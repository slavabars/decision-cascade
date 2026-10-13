"""Выбор устройства и типа вычислений для энкодеров.

Пользователь задаёт устройство в момент запуска через окружение:

    DECISION_DEVICE=auto|cpu|cuda|cuda:0|mps   (по умолчанию auto)
    DECISION_DTYPE=float32|float16|bfloat16     (по умолчанию float32)

``auto`` выбирает CUDA, если она доступна, иначе CPU, — один и тот же образ
работает и на CPU-машине, и на GPU-хосте без правки конфигов. Явно запрошенное
недоступное устройство — это ошибка, а не тихий откат на CPU.
"""
from __future__ import annotations

import torch

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(spec: str | None) -> str:
    """Преобразует спецификацию устройства из окружения в строку torch.

    ``None``/``""``/``auto`` -> ``cuda`` при наличии, иначе ``cpu``. Явно
    запрошенная, но недоступная CUDA/MPS — ошибка (не молчаливый откат).
    """
    raw = (spec or "auto").strip().lower()
    if raw in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(raw)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"DECISION_DEVICE={spec!r} is not a valid device"
        ) from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"DECISION_DEVICE={spec!r} requested but CUDA is not available"
            )
        count = torch.cuda.device_count()
        index = device.index if device.index is not None else 0
        if index >= count:
            raise RuntimeError(
                f"DECISION_DEVICE={spec!r} but only {count} CUDA device(s) available"
            )
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"DECISION_DEVICE={spec!r} requested but MPS is not available"
        )
    return str(device)


def resolve_dtype(spec: str | None) -> torch.dtype:
    """Разбирает тип вычислений; неизвестный тип — ошибка с подсказкой."""
    raw = (spec or "float32").strip().lower()
    if raw not in _DTYPES:
        raise RuntimeError(
            f"DECISION_DTYPE must be one of {sorted(_DTYPES)}, got {spec!r}"
        )
    return _DTYPES[raw]


def check_dtype_device(dtype: torch.dtype, device: str) -> None:
    """Отсекает заведомо несовместимые пары (float16 на CPU не работает)."""
    if device == "cpu" and dtype is torch.float16:
        raise RuntimeError(
            "DECISION_DTYPE=float16 is not supported on CPU; "
            "use float32 or bfloat16"
        )
