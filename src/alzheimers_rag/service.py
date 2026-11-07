"""Compatibility imports; service implementation lives in the API application package."""

from .api.config import ServiceConfig
from .api.services.query import QueryService, ServiceBusyError, ServiceRequestError, open_service

__all__ = [
    "QueryService",
    "ServiceBusyError",
    "ServiceConfig",
    "ServiceRequestError",
    "open_service",
]
