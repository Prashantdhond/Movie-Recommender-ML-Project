import axios from "axios";

export const API_BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export async function getHome(category, limit = 24) {
  const { data } = await api.get("/home", {
    params: { category, limit },
  });
  return data;
}

export async function searchMovies(query, page = 1) {
  const { data } = await api.get("/tmdb/search", {
    params: { query, page },
  });
  return data.results || [];
}

export async function getMovieDetails(tmdbId) {
  const { data } = await api.get(`/movie/id/${tmdbId}`);
  return data;
}

export async function getMovieBundle(query) {
  const { data } = await api.get("/movie/search", {
    params: {
      query,
      tfidf_top_n: 12,
      genre_limit: 12,
    },
  });
  return data;
}

export async function checkHealth() {
  const { data } = await api.get("/health");
  return data;
}
