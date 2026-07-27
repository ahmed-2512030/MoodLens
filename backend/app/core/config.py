from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MoodLens API"
    version: str = "0.1.0"

    # Model(s) served. One or more GoEmotions-trained models; when more than one
    # is given they are ENSEMBLED (Ekman scores averaged). The two-model default
    # is the benchmark winner (macro-F1 0.673 > 0.663 single, PROJECT_REPORT #7).
    # Override with the MODEL_NAMES env var, e.g. a JSON list, or one model for
    # lower latency: MODEL_NAMES='["bhadresh-savani/bert-base-go-emotion"]'
    model_names: list[str] = [
        "bhadresh-savani/bert-base-go-emotion",
        "SamLowe/roberta-base-go_emotions",
    ]

    # Return the top-k fine-grained labels alongside the 6 Ekman classes.
    top_k: int = 5

    # Max rows accepted from an uploaded CSV/JSON in one request.
    max_rows: int = 5000

    # CORS origins for the Next.js frontend.
    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
