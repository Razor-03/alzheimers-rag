"""Compose the FastAPI application; request handling lives in routers."""

from fastapi import FastAPI

from .. import __version__
from .config import ServiceConfig
from .exception_handlers import register_exception_handlers
from .lifespan import build_lifespan
from .routers import health, query
from .services.query import open_service


def create_app(*, config: ServiceConfig | None = None, service_factory=open_service) -> FastAPI:
    app = FastAPI(
        title="Alzheimer's Research RAG",
        version=__version__,
        lifespan=build_lifespan(config=config, service_factory=service_factory),
        description="Research answers with evidence provenance. Citation support requires review.",
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(query.router)
    return app
