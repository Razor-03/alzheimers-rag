"""HTTP application package; the public factory keeps API dependencies optional."""


def create_app(**kwargs):
    """Compatibility entry point for alzheimers_rag.api:create_app."""
    from .main import create_app as factory

    return factory(**kwargs)
