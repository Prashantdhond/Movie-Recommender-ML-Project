import os
import time
import pickle
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

logger = logging.getLogger("uvicorn.error")


# =========================
# GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
KMEANS_PATH = os.path.join(BASE_DIR, "kmeans_model.pkl")
KNN_PATH = os.path.join(BASE_DIR, "knn_model.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
kmeans_obj: Any = None  # trained sklearn KMeans
knn_obj: Any = None  # trained sklearn NearestNeighbors (cosine, brute)
CLUSTER_LABELS: Optional[np.ndarray] = None  # K-Means cluster id per row of df
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# Shared HTTP client (created once at startup)
http_client: Optional[httpx.AsyncClient] = None

# Small in-memory TTL cache: key -> (timestamp, data)
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL = 300  # seconds a cached response is considered fresh
CACHE_MAX_ITEMS = 500  # stale entries are kept as fallback, up to this many

MAX_RETRIES = 3

# Recommender: how many nearest neighbours KNN fetches before the cluster filter is applied
CANDIDATE_POOL = 1000


# =========================
# LIFESPAN (startup / shutdown)
# =========================
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    global kmeans_obj, knn_obj, CLUSTER_LABELS

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    with open(KMEANS_PATH, "rb") as f:
        kmeans_obj = pickle.load(f)

    with open(KNN_PATH, "rb") as f:
        knn_obj = pickle.load(f)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    if "cluster" not in df.columns:
        raise RuntimeError(
            "df.pkl has no 'cluster' column. Re-run movies.ipynb to regenerate the models."
        )

    if not (len(df) == tfidf_matrix.shape[0] == len(kmeans_obj.labels_)):
        raise RuntimeError(
            "df.pkl, tfidf_matrix.pkl and kmeans_model.pkl have different sizes. "
            "They must all come from the same run of movies.ipynb."
        )

    CLUSTER_LABELS = df["cluster"].to_numpy()
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client

    load_pickles()

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(20.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        headers={"Accept": "application/json"},
    )
    logger.info(
        "Startup complete: TF-IDF + K-Means (K=%d) + KNN loaded, HTTP client ready",
        int(kmeans_obj.n_clusters),
    )

    yield

    if http_client is not None:
        await http_client.aclose()
        http_client = None


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    # Name kept for frontend compatibility; items now come from K-Means + KNN.
    title: str
    score: float
    cluster: Optional[int] = None
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]
    query_cluster: Optional[int] = None
    recommendation_method: str = "kmeans+knn"


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


def _cache_put(key: str, data: Dict[str, Any]) -> None:
    if len(_CACHE) >= CACHE_MAX_ITEMS:
        # drop the oldest entry
        oldest_key = min(_CACHE, key=lambda k: _CACHE[k][0])
        _CACHE.pop(oldest_key, None)
    _CACHE[key] = (time.time(), data)


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resilient TMDB GET:
    - fresh cache hit -> return immediately
    - retries with backoff on network errors / 429 / 5xx
    - if all retries fail but we have a stale cached copy -> serve it
    - 404 from TMDB -> 404, other failures -> 502
    """
    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")

    cache_key = f"{path}|{sorted((k, str(v)) for k, v in params.items())}"
    hit = _CACHE.get(cache_key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    last_err = "unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await http_client.get(f"{TMDB_BASE}{path}", params=q)

            if r.status_code == 200:
                data = r.json()
                _cache_put(cache_key, data)
                return data

            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"TMDB status {r.status_code}"
                logger.warning(
                    "TMDB attempt %d/%d for %s -> %s",
                    attempt, MAX_RETRIES, path, last_err,
                )
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        await asyncio.sleep(min(int(retry_after), 5))
            elif r.status_code == 404:
                raise HTTPException(status_code=404, detail="TMDB resource not found")
            else:
                # 401 (bad key), 400, etc. won't fix themselves: don't retry
                raise HTTPException(
                    status_code=502,
                    detail=f"TMDB error {r.status_code}: {r.text}",
                )

        except httpx.RequestError as e:
            last_err = f"{type(e).__name__}: {e!r}"
            logger.warning(
                "TMDB attempt %d/%d for %s failed -> %s",
                attempt, MAX_RETRIES, path, last_err,
            )

        if attempt < MAX_RETRIES:
            await asyncio.sleep(0.5 * attempt)  # backoff: 0.5s, 1.0s

    # All retries failed: fall back to stale cache if available
    if hit:
        logger.warning("Serving STALE cache for %s (%s)", path, last_err)
        return hit[1]

    raise HTTPException(status_code=502, detail=f"TMDB unreachable: {last_err}")


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """Raw TMDB response for keyword search (MULTIPLE results)."""
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF HELPERS
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be a dict(title -> index) or a pandas Series
    (index=title, value=index). Normalized into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
    except Exception:
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )
    return title_to_idx


def get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from the local df using cosine similarity
    on the TF-IDF matrix (rows are assumed L2-normalized, as TfidfVectorizer does).
    """
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]
    product = tfidf_matrix @ qv.T
    scores = product.toarray().ravel() if hasattr(product, "toarray") else np.ravel(product)

    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        i = int(i)
        if i == int(idx):
            continue
        try:
            title_i = str(df.iloc[i]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[i])))
        if len(out) >= top_n:
            break
    return out


def ml_recommend_titles(
    query_title: str, top_n: int = 10, pool: int = CANDIDATE_POOL
) -> Tuple[int, List[Tuple[str, float, int]]]:
    """
    K-Means + KNN recommender (the trained-model pipeline):

      1. find the K-Means cluster of the selected movie
      2. ask the saved KNN model (cosine, brute) for its nearest `pool` movies
      3. keep only neighbours from the same cluster, ordered by cosine distance
      4. return the top_n; if the cluster is too small, top up with the nearest
         movies from other clusters so the caller always gets top_n results

    Returns (query_cluster, [(title, cosine_similarity, cluster_id), ...]).
    """
    if (
        df is None
        or tfidf_matrix is None
        or knn_obj is None
        or CLUSTER_LABELS is None
    ):
        raise HTTPException(status_code=500, detail="ML models not loaded")

    idx = get_local_idx_by_title(query_title)
    n_docs = tfidf_matrix.shape[0]
    pool = max(1, min(pool, n_docs - 1))

    dist, ind = knn_obj.kneighbors(tfidf_matrix[idx], n_neighbors=pool + 1)
    ind, dist = ind[0], dist[0]

    keep = ind != idx  # never recommend the selected movie itself
    ind, dist = ind[keep], dist[keep]

    query_cluster = int(CLUSTER_LABELS[idx])
    same_cluster = CLUSTER_LABELS[ind] == query_cluster

    chosen = list(np.flatnonzero(same_cluster)[:top_n])
    if len(chosen) < top_n:
        others = np.flatnonzero(~same_cluster)
        chosen += list(others[: top_n - len(chosen)])

    out: List[Tuple[str, float, int]] = []
    for p in chosen:
        row = int(ind[p])
        out.append(
            (str(df.iloc[row]["title"]), float(1.0 - dist[p]), int(CLUSTER_LABELS[row]))
        )
    return query_cluster, out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title.
    Returns None on any failure (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception as e:
        logger.warning("Could not attach TMDB card for '%s': %r", title, e)
        return None


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    category:
      - trending (trending/movie/day)
      - popular, top_rated, upcoming, now_playing  (movie/{category})
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Home route failed")
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """Returns RAW TMDB shape with a 'results' list."""
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Given a TMDB movie ID: fetch details, pick first genre,
    discover popular movies in that genre.
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- K-MEANS + KNN (trained ML models) ----------
@app.get("/recommend/ml")
async def recommend_ml(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    """Selected movie -> its K-Means cluster -> KNN ranking inside that cluster."""
    cluster, recs = ml_recommend_titles(title, top_n=top_n)
    return {
        "query": title,
        "cluster": cluster,
        "method": "kmeans+knn",
        "recommendations": [
            {"title": t, "score": s, "cluster": c} for t, s, c in recs
        ],
    }


# ---------- MODEL INFO (handy for the viva demo) ----------
@app.get("/model/info")
def model_info():
    sizes = df["cluster"].value_counts().sort_index()
    return {
        "n_movies": int(len(df)),
        "tfidf_features": int(tfidf_matrix.shape[1]),
        "kmeans_k": int(kmeans_obj.n_clusters),
        "knn_metric": str(knn_obj.metric),
        "knn_algorithm": str(knn_obj.algorithm),
        "cluster_sizes": {int(k): int(v) for k, v in sizes.items()},
    }


# ---------- COSINE BASELINE (TF-IDF only, no trained model) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    Selects the BEST TMDB match for the query and returns:
      - movie details
      - K-Means + KNN recommendations (local) + posters
      - Genre recommendations (TMDB) + posters
    For MULTIPLE matches, use /tmdb/search.
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) K-Means + KNN recommendations (never crash endpoint)
    recs: List[Tuple[str, float, int]] = []
    query_cluster: Optional[int] = None
    for name in (details.title, query):
        try:
            query_cluster, recs = ml_recommend_titles(name, top_n=tfidf_top_n)
            break
        except Exception:
            continue

    # Fetch posters concurrently, but limit parallelism to be gentle on TMDB
    sem = asyncio.Semaphore(5)

    async def _card(title: str) -> Optional[TMDBMovieCard]:
        async with sem:
            return await attach_tmdb_card_by_title(title)

    cards_for_recs = await asyncio.gather(*[_card(t) for t, _, _ in recs])

    tfidf_items: List[TFIDFRecItem] = [
        TFIDFRecItem(title=t, score=s, cluster=c, tmdb=card)
        for (t, s, c), card in zip(recs, cards_for_recs)
    ]

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
        query_cluster=query_cluster,
    )
