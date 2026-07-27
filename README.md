# MoodLens

A BERT-based emotion & sentiment analysis platform for real-time public opinion
intelligence. Monorepo: FastAPI backend + Next.js frontend.

## Architecture

```
MoodLens/
├── backend/          FastAPI + HuggingFace Transformers
│   ├── app/
│   │   ├── core/         config + GoEmotions→Ekman mapping
│   │   ├── services/     model loader, classifier, aggregation
│   │   ├── routers/      /analyze /upload /report /health
│   │   ├── schemas.py    Pydantic request/response models
│   │   └── main.py       FastAPI app (loads model once at startup)
│   └── requirements.txt
├── frontend/         Next.js 15 (App Router, TS, Tailwind, Recharts)
│   ├── app/          dashboard page
│   ├── components/   emotion charts
│   └── lib/api.ts    typed API client
└── docker-compose.yml
```

**Model:** an **ensemble** of two GoEmotions-trained transformers —
`bhadresh-savani/bert-base-go-emotion` + `SamLowe/roberta-base-go_emotions` — whose
28-label outputs are aggregated to the 6 Ekman classes (joy, anger, sadness, fear,
surprise, disgust) + neutral and averaged. The ensemble beats either model alone
(macro-F1 0.673 vs 0.663; see `PROJECT_REPORT.md`). Change via the `MODEL_NAMES` env
var — pass a single-model JSON list for lower latency.

## API

| Method | Path       | Purpose                                   |
|--------|------------|-------------------------------------------|
| GET    | `/health`  | status + which model is loaded            |
| POST   | `/analyze` | single text → emotion scores              |
| POST   | `/upload`  | CSV/JSON batch → per-row results + summary |
| POST   | `/report`  | batch summary → downloadable PDF          |

Interactive docs at `http://localhost:8000/docs`.

## Run — Docker (both services)

```bash
docker compose up --build
# frontend: http://localhost:3000   backend: http://localhost:8000
```

## Run — local dev

**Backend**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload     # first start downloads the model (~400MB)
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

Try `backend/sample_data.csv` in the upload panel.
