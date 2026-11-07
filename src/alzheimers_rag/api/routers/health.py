"""Worker liveness and database readiness endpoints."""

from fastapi import APIRouter, HTTPException

from ..dependencies import ServiceDependency
from ..schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"description": "Database unavailable"}},
)
def ready(service: ServiceDependency):
    if not service.is_ready():
        raise HTTPException(503, "Query service is not ready")
    return {"status": "ready", **service.capabilities}
