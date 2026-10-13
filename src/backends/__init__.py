"""Бэкенды принятия решений: локальные головы и внешние SystemOne-сервисы."""
from src.backends.base import (
    Backend,
    BackendResult,
    Capabilities,
    LocalEngine,
    missing_modalities,
    requested_modalities,
)
from src.backends.http import HttpSystemOneBackend, parse_modalities
from src.backends.local import LocalHeadsBackend

__all__ = [
    "Backend",
    "BackendResult",
    "Capabilities",
    "HttpSystemOneBackend",
    "LocalEngine",
    "LocalHeadsBackend",
    "missing_modalities",
    "parse_modalities",
    "requested_modalities",
]
