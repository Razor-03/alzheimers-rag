"""Own shared resources for the duration of an application worker."""

from contextlib import asynccontextmanager

from .config import ServiceConfig
from .services.query import open_service


def build_lifespan(*, config: ServiceConfig | None = None, service_factory=open_service):
    @asynccontextmanager
    async def lifespan(app):
        settings = config if config is not None else ServiceConfig.from_env()
        with service_factory(settings) as service:
            app.state.service = service
            try:
                yield
            finally:
                app.state.service = None

    return lifespan
