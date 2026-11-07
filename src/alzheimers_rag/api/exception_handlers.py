"""Map query failures to sanitized HTTP responses in one place."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..generation import GenerationError
from .services.query import ServiceBusyError, ServiceRequestError

logger = logging.getLogger(__name__)


class QueryBackendError(Exception):
    def __init__(self, error_type: str):
        super().__init__("A query backend is unavailable or failed")
        self.error_type = error_type


async def request_error(request: Request, exc: ServiceRequestError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def busy_error(request: Request, exc: ServiceBusyError):
    return JSONResponse(
        status_code=429,
        content={"detail": "Query service is busy; retry later"},
        headers={"Retry-After": "1"},
    )


async def generation_error(request: Request, exc: GenerationError):
    return JSONResponse(
        status_code=502, content={"detail": "Model output failed answer or citation validation"}
    )


async def backend_error(request: Request, exc: QueryBackendError):
    # No raw exception message, traceback, evidence or credentials in responses/logs.
    logger.warning("Query backend failed (%s)", exc.error_type)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def register_exception_handlers(app: FastAPI):
    app.add_exception_handler(ServiceRequestError, request_error)
    app.add_exception_handler(ServiceBusyError, busy_error)
    app.add_exception_handler(GenerationError, generation_error)
    app.add_exception_handler(QueryBackendError, backend_error)
