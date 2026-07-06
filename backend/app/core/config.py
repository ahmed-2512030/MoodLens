from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MoodLens API"
    version: str = "0.1.0"

    # HuggingFace model. BERT fine-tuned on GoEmotions (28 labels).
    # Swap freely, e.g. SamLowe/roberta-base-go_emotions for higher accuracy.
    model_name: str = "bhadresh-savani/bert-base-go-emotion"

    # Return the top-k fine-grained labels alongside the 6 Ekman classes.
    top_k: int = 5

    # Max rows accepted from an uploaded CSV/JSON in one request.
    max_rows: int = 5000

    # CORS origins for the Next.js frontend.
    cors_origins: list[str] = ["http://localhost:3000"]


settings = Settings()
