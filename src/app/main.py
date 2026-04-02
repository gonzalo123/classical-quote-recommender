"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.routes import router
from app.utils.logging import configure_logging
from settings import APP_NAME, LOG_LEVEL

configure_logging(LOG_LEVEL)

app = FastAPI(
    title=APP_NAME,
    version="0.1.0",
    description="Rhetorical classical quote recommender PoC.",
)
app.include_router(router)
