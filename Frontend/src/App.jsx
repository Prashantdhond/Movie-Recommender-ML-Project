import React, { useEffect, useState } from "react";
import {
  getHome,
  searchMovies,
  getMovieBundle,
  getMovieDetails,
} from "./api";

const CATEGORIES = [
  { key: "trending", label: "Trending" },
  { key: "popular", label: "Popular" },
  { key: "top_rated", label: "Top Rated" },
  { key: "now_playing", label: "Now Playing" },
  { key: "upcoming", label: "Upcoming" },
];

function MovieCard({ movie, onClick }) {
  return (
    <button className="movie-card" onClick={() => onClick(movie)}>
      <div className="poster-wrap">
        {movie.poster_url ? (
          <img src={movie.poster_url} alt={movie.title} loading="lazy" />
        ) : (
          <div className="poster-empty">No Poster</div>
        )}

        {movie.vote_average != null && (
          <span className="rating">
            ★ {Number(movie.vote_average).toFixed(1)}
          </span>
        )}
      </div>

      <div className="movie-card-info">
        <h3>{movie.title}</h3>
        <p>{movie.release_date?.slice(0, 4) || "N/A"}</p>
      </div>
    </button>
  );
}

function Section({ title, movies, onMovieClick }) {
  if (!movies?.length) return null;

  return (
    <section className="section">
      <div className="section-heading">
        <h2>{title}</h2>
        <span>{movies.length} movies</span>
      </div>

      <div className="movie-grid">
        {movies.map((movie) => (
          <MovieCard
            key={`${movie.tmdb_id}-${movie.title}`}
            movie={movie}
            onClick={onMovieClick}
          />
        ))}
      </div>
    </section>
  );
}

function DetailsModal({ selected, onClose, onMovieClick }) {
  const [details, setDetails] = useState(selected?.details || null);
  const [bundle, setBundle] = useState(selected?.bundle || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!selected) return;

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError("");
      setDetails(selected.details || null);
      setBundle(selected.bundle || null);

      try {
        // The bundle endpoint connects:
        // TMDB details + local TF-IDF recommendations + genre recommendations.
        const data = await getMovieBundle(selected.movie.title);
        if (!cancelled) {
          setBundle(data);
          setDetails(data.movie_details);
        }
      } catch (err) {
        // Fallback to the direct details endpoint.
        try {
          const data = await getMovieDetails(selected.movie.tmdb_id);
          if (!cancelled) setDetails(data);
        } catch {
          if (!cancelled) {
            setError(
              err?.response?.data?.detail ||
                "Could not load movie details. Check that FastAPI is running."
            );
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [selected]);

  if (!selected) return null;

  const tfidfMovies =
    bundle?.tfidf_recommendations
      ?.map((item) => item.tmdb)
      .filter(Boolean) || [];

  const genreMovies = bundle?.genre_recommendations || [];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-btn" onClick={onClose}>
          ×
        </button>

        {loading && !details ? (
          <div className="modal-loading">Loading movie...</div>
        ) : details ? (
          <>
            <div
              className="details-hero"
              style={{
                backgroundImage: details.backdrop_url
                  ? `linear-gradient(90deg, rgba(5,5,5,.98), rgba(5,5,5,.68), rgba(5,5,5,.2)), url(${details.backdrop_url})`
                  : undefined,
              }}
            >
              <div className="details-content">
                <img
                  className="details-poster"
                  src={details.poster_url}
                  alt={details.title}
                />

                <div>
                  <div className="eyebrow">MOVIE DETAILS</div>
                  <h1>{details.title}</h1>

                  <div className="meta">
                    <span>{details.release_date || "Release date N/A"}</span>
                    {details.genres?.map((genre) => (
                      <span className="pill" key={genre.id}>
                        {genre.name}
                      </span>
                    ))}
                  </div>

                  <p className="overview">
                    {details.overview || "No overview available."}
                  </p>
                </div>
              </div>
            </div>

            {error && <div className="error-box">{error}</div>}

            <div className="recommendation-area">
              <Section
                title="Because You Watched This"
                movies={tfidfMovies}
                onMovieClick={onMovieClick}
              />

              <Section
                title="More From This Genre"
                movies={genreMovies}
                onMovieClick={onMovieClick}
              />
            </div>
          </>
        ) : (
          <div className="modal-loading">No details available.</div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [activeCategory, setActiveCategory] = useState("trending");
  const [movies, setMovies] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [query, setQuery] = useState("");
  const [searchMode, setSearchMode] = useState(false);
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);

  async function loadCategory(category) {
    setLoading(true);
    setError("");
    setSearchMode(false);
    setSearchResults([]);

    try {
      const data = await getHome(category, 24);
      setMovies(data);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          "Cannot connect to FastAPI. Start uvicorn on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadCategory(activeCategory);
  }, [activeCategory]);

  async function handleSearch(e) {
    e.preventDefault();
    const value = query.trim();

    if (!value) {
      loadCategory(activeCategory);
      return;
    }

    setSearching(true);
    setError("");
    setSearchMode(true);

    try {
      const data = await searchMovies(value);
      setSearchResults(data);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          "Search failed. Make sure FastAPI and your TMDB API key are working."
      );
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  }

  function handleMovieClick(movie) {
    setSelected({ movie });
  }

  const displayedMovies = searchMode ? searchResults : movies;

  return (
    <div className="app">
      <header className="navbar">
        <a
          className="brand"
          href="#top"
          onClick={() => {
            setSearchMode(false);
            setQuery("");
          }}
        >
          <span className="brand-mark">C</span>
          <span>Cine<span>Match</span></span>
        </a>

        <nav>
          {CATEGORIES.slice(0, 4).map((category) => (
            <button
              key={category.key}
              className={
                !searchMode && activeCategory === category.key
                  ? "nav-link active"
                  : "nav-link"
              }
              onClick={() => setActiveCategory(category.key)}
            >
              {category.label}
            </button>
          ))}
        </nav>

        <form className="search" onSubmit={handleSearch}>
          <span>⌕</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search movies..."
          />
        </form>
      </header>

      <main id="top">
        {!searchMode && (
          <section className="hero">
            <div className="hero-copy">
              <div className="eyebrow">AI MOVIE RECOMMENDATION SYSTEM</div>
              <h1>Find your next<br /><em>favorite movie.</em></h1>
              <p>
                Search a movie and discover similar titles using your
                TF-IDF + cosine similarity model, with movie data powered
                by TMDB.
              </p>

              <div className="hero-actions">
                <button
                  className="primary-btn"
                  onClick={() => {
                    document
                      .getElementById("movie-section")
                      ?.scrollIntoView({ behavior: "smooth" });
                  }}
                >
                  Explore Movies
                </button>
                <button
                  className="secondary-btn"
                  onClick={() => {
                    setActiveCategory("trending");
                    setSearchMode(false);
                  }}
                >
                  Trending Now
                </button>
              </div>
            </div>

            <div className="hero-orbit">
              <div className="hero-circle">
                <span>★</span>
              </div>
            </div>
          </section>
        )}

        <div className="category-tabs">
          {CATEGORIES.map((category) => (
            <button
              key={category.key}
              className={
                !searchMode && activeCategory === category.key
                  ? "category-tab active"
                  : "category-tab"
              }
              onClick={() => setActiveCategory(category.key)}
            >
              {category.label}
            </button>
          ))}
        </div>

        <section className="content" id="movie-section">
          {searchMode ? (
            <>
              <div className="section-heading">
                <div>
                  <div className="eyebrow">SEARCH RESULTS</div>
                  <h2>Results for “{query}”</h2>
                </div>
                <button
                  className="clear-search"
                  onClick={() => {
                    setQuery("");
                    setSearchMode(false);
                    loadCategory(activeCategory);
                  }}
                >
                  Clear
                </button>
              </div>

              {searching ? (
                <LoadingGrid />
              ) : displayedMovies.length ? (
                <div className="movie-grid">
                  {displayedMovies.map((movie) => (
                    <MovieCard
                      key={movie.tmdb_id}
                      movie={movie}
                      onClick={handleMovieClick}
                    />
                  ))}
                </div>
              ) : (
                <EmptyState text="No movies found." />
              )}
            </>
          ) : (
            <>
              <div className="section-heading">
                <div>
                  <div className="eyebrow">DISCOVER</div>
                  <h2>
                    {CATEGORIES.find((c) => c.key === activeCategory)?.label}
                  </h2>
                </div>
                <span>{displayedMovies.length} titles</span>
              </div>

              {loading ? (
                <LoadingGrid />
              ) : displayedMovies.length ? (
                <div className="movie-grid">
                  {displayedMovies.map((movie) => (
                    <MovieCard
                      key={movie.tmdb_id}
                      movie={movie}
                      onClick={handleMovieClick}
                    />
                  ))}
                </div>
              ) : (
                <EmptyState text="No movies available." />
              )}
            </>
          )}

          {error && <div className="error-box">{error}</div>}
        </section>
      </main>

      <footer>
        <div>
          <strong>CineMatch</strong>
          <p>AI-powered movie discovery using TMDB + TF-IDF.</p>
        </div>
        <span>Movie Recommendation System</span>
      </footer>

      <DetailsModal
        selected={selected}
        onClose={() => setSelected(null)}
        onMovieClick={handleMovieClick}
      />
    </div>
  );
}

function LoadingGrid() {
  return (
    <div className="movie-grid">
      {Array.from({ length: 12 }).map((_, index) => (
        <div className="skeleton-card" key={index}>
          <div className="skeleton-poster" />
          <div className="skeleton-line" />
          <div className="skeleton-line small" />
        </div>
      ))}
    </div>
  );
}

function EmptyState({ text }) {
  return <div className="empty-state">{text}</div>;
}