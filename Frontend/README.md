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


Requirements · TXT
# --- API (main.py) ---
fastapi>=0.110
uvicorn[standard]>=0.29
httpx>=0.27
python-dotenv>=1.0
pydantic>=2.6
 
# --- ML / notebook (movies.ipynb) ---
numpy>=1.26
pandas>=2.2
scipy>=1.11
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
nltk>=3.8
ipykernel>=6.29
jupyter>=1.0
 

