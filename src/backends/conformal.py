"""Conformal-каскад: порог на ``P(ошибка)`` из сертификата с гарантией.

Отличие от risk-aware каскада — не механика эскалации, а откуда берётся порог.
Risk-каскад принимает бюджет ошибок как *желание*; conformal-каскад берёт порог
из ``conformal.json``, посчитанного на отложенной калибровке: с вероятностью
``1 - delta`` доля ошибок среди принятых решений не выше ``alpha``
(см. ``src.conformal``). Сертификат несёт и вердикт гейта «не навреди»: если на
тесте порог хуже простого ранжирования по уверенности, каскад фолбэчит на
``DECISION_CASCADE_THRESHOLD`` и пишет в ``usage`` причину ``conformal_gate``.

Обратную связь такой каскад не принимает: сертификат — свойство калибровочной
выборки, а не поток примеров, и «дообучение» порога без пересчёта границы
сломало бы обещание.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.backends.cascade import DEFAULT_THRESHOLD, RiskAwareCascadeBackend
from src.conformal import load_certificate, validate_certificate


class ConformalCascadeBackend(RiskAwareCascadeBackend):
    """Каскад с порогом из conformal-сертификата и жёстким гейтом."""

    def __init__(
        self,
        primary,
        fallback,
        risk_model,
        certificate: dict[str, Any] | str | Path,
        *,
        fallback_threshold: float = DEFAULT_THRESHOLD,
    ):
        if isinstance(certificate, (str, Path)):
            certificate = load_certificate(certificate)
        certificate = validate_certificate(dict(certificate))
        threshold = float(certificate["threshold"])
        gate = certificate.get("gate") or {"accepted": True, "worse": []}
        super().__init__(
            primary,
            fallback,
            risk_model,
            risk_budget=max(threshold, 0.0),
            gate=gate,
            fallback_threshold=fallback_threshold,
            gate_delta=0.0,
            few_shot_n=None,
        )
        self.certificate = certificate
        self.certificate_threshold = threshold
        self.name = f"conformal-cascade[{primary.name}->{fallback.name}]"

    @property
    def escalation_reason(self) -> str | None:
        if self.gate_rejected:
            return "conformal_gate"
        return "conformal"

    @property
    def policy(self) -> str:
        """Имя политики для ``/info``: сертификат нельзя спутать с risk-бюджетом."""
        return "conformal"

    def _escalate(self, model: Any, decision: Any) -> bool:
        if self.gate_rejected:
            # сертификат отклонён: работает лучший baseline — порог уверенности
            return decision.confidence < self.fallback_threshold
        return float(model.predict_error(decision)) > self.certificate_threshold

    def conformal_info(self) -> dict[str, Any]:
        """Параметры сертификата для ``GET /info -> engine``.

        Ключи — часть контракта 1.x: их можно добавлять, но не переименовывать
        и не убирать. ``conformal_test`` — срез гейта на отложенном тесте, если
        сертификат его нёс.
        """
        certificate = self.certificate
        info: dict[str, Any] = {
            "policy": self.policy,
            "conformal_alpha": float(certificate["alpha"]),
            "conformal_delta": float(certificate["delta"]),
            "conformal_threshold": self.certificate_threshold,
            "conformal_coverage": float(certificate["coverage"]),
            "conformal_risk": certificate.get("empirical_risk"),
            "conformal_upper_bound": float(certificate["upper_bound"]),
            "conformal_calibration_n": int(certificate["n"]),
            "conformal_gate": "rejected" if self.gate_rejected else "ok",
            "conformal_gate_worse": list(self.gate_worse),
        }
        gate = certificate.get("gate")
        if isinstance(gate, dict) and isinstance(gate.get("test"), dict):
            info["conformal_test"] = dict(gate["test"])
        return info
