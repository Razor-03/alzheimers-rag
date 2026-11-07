"""HTTP adapter for the shared research-answer service."""

from fastapi import APIRouter

from ...config import RerankConfig, RetrievalConfig
from ...generation import GenerationError
from ..dependencies import ServiceDependency
from ..exception_handlers import QueryBackendError
from ..schemas.query import AnswerResponse, QueryRequest
from ..services.query import ServiceBusyError, ServiceRequestError

router = APIRouter(tags=["research"])


@router.post(
    "/query",
    response_model=AnswerResponse,
    responses={
        429: {"description": "This worker is already handling a query; retry later"},
        502: {"description": "Model output failed answer/citation validation"},
        503: {"description": "A query backend is unavailable or failed"},
    },
)
def query(payload: QueryRequest, service: ServiceDependency):
    try:
        return service.answer(
            payload.query,
            RetrievalConfig(payload.top_k, payload.alpha, payload.section_type, payload.keyword),
            rerank_config=RerankConfig(
                candidate_pool_size=payload.candidate_pool_size, rerank_top_k=payload.top_k
            )
            if payload.rerank
            else None,
            context=payload.context,
        )
    except (ServiceRequestError, ServiceBusyError, GenerationError):
        raise
    except Exception as exc:
        raise QueryBackendError(type(exc).__name__) from None
