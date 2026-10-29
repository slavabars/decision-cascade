"""Типизированные ошибки уровня сервиса.

Сервис поднимает их там, где раньше было «тихое» поведение, а API отображает в
явные коды ответа (см. ``api/server/main.py``).
"""
from __future__ import annotations


class UntrainedDecisionError(RuntimeError):
    """Схема вопросов без обученного чекпоинта, а политика требует отказа.

    Возникает, когда политика ``DECISION_UNTRAINED`` равна ``refuse`` или
    ``fallback``, но под подпись набора вопросов нет весов. Отдавать решения со
    случайно инициализированных голов нельзя: их ``confidence`` ничего не значит,
    и каскад может принять по нему неверное решение об эскалации.
    """

    def __init__(self, signature: str, checkpoint_dir: str | None = None) -> None:
        self.signature = signature
        self.checkpoint_dir = checkpoint_dir
        hint = (
            f"put a trained {signature}.pt into {checkpoint_dir}"
            if checkpoint_dir
            else "set DECISION_CHECKPOINT_DIR with trained heads"
        )
        super().__init__(
            f"no trained checkpoint for questions signature {signature!r}; "
            f"refusing to serve untrained heads ({hint})"
        )


class RemoteBackendError(RuntimeError):
    """Внешний SystemOne-бэкенд недоступен или ответил ошибкой после retry.

    ``message`` не содержит тела запроса (state/images) — только имя бэкенда и
    код ответа: тексты состояний и изображения не должны попадать в логи и в
    ответы об ошибках.
    """

    def __init__(
        self,
        message: str,
        *,
        backend: str = "remote",
        status_code: int | None = None,
    ) -> None:
        self.backend = backend
        self.status_code = status_code
        super().__init__(message)
