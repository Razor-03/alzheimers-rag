"""Request dependencies available to routers and test dependency overrides."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from .services.query import QueryService


def get_service(request: Request) -> QueryService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "Query service is not initialized")
    return service


ServiceDependency = Annotated[QueryService, Depends(get_service)]
