# CineMatch React Frontend

React/Vite frontend connected to the FastAPI movie recommendation backend.

## 1. Start FastAPI

From your `Model` folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload
```

FastAPI should be available at:

`http://127.0.0.1:8000`

## 2. Install frontend packages

Open another PowerShell terminal and enter the `Frontend` folder:

```powershell
npm install
```

## 3. Start React

```powershell
npm run dev
```

Open the Vite URL shown in the terminal, normally:

`http://localhost:5173`

## Connected FastAPI endpoints

- `GET /home?category=trending&limit=24`
- `GET /home?category=popular&limit=24`
- `GET /home?category=top_rated&limit=24`
- `GET /home?category=now_playing&limit=24`
- `GET /home?category=upcoming&limit=24`
- `GET /tmdb/search?query=...`
- `GET /movie/id/{tmdb_id}`
- `GET /movie/search?query=...`
- `GET /recommend/genre?tmdb_id=...`
- `GET /recommend/tfidf?title=...`

The `/movie/search` endpoint is used when a movie is opened. It returns the selected movie, TF-IDF recommendations and genre recommendations.
