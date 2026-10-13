"""Маршрутизация обращения по решениям decision-cascade.

Показывает типичную интеграцию «мозга решений» в чат-агента: по типизированным
решениям (noul/choice/score) выбирается команда, приоритет, эскалация и тон
ответа. Логика чистая (без сети) и легко тестируется отдельно от SDK.

Каскад: ``route_ticket`` решает, нужна ли эскалация; вызывающий код (см.
``examples/integrations/chat_agent.py``) может на этом основании переспросить внешний сервис.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRIORITIES = ("low", "normal", "high", "urgent")


@dataclass(frozen=True)
class RoutingPolicy:
    """Пороги, по которым решения превращаются в действие агента."""

    escalate_threshold: float = 0.5
    urgent_threshold: float = 0.6
    high_threshold: float = 0.3
    churn_threshold: float = 3.0
    heat_urgent: float = 3.5
    heat_high: float = 2.0
    default_team: str = "общая поддержка"


@dataclass(frozen=True)
class AgentAction:
    team: str
    priority: str
    escalate: bool
    tone: str
    reply: str
    rationale: tuple[str, ...] = ()


def _noul(decisions: dict[str, Any], key: str) -> float | None:
    d = decisions.get(key)
    if isinstance(d, dict) and d.get("type") == "noul":
        return float(d["noul"])
    return None


def _choice(decisions: dict[str, Any], key: str) -> str | None:
    d = decisions.get(key)
    if isinstance(d, dict) and d.get("type") == "choice" and "choice" in d:
        return str(d["choice"])
    return None


def _score(decisions: dict[str, Any], key: str) -> float | None:
    d = decisions.get(key)
    if isinstance(d, dict) and d.get("type") == "score":
        return float(d["score"])
    return None


def _priority(
    urgency: float,
    heat: float,
    churn: float,
    policy: RoutingPolicy,
    rationale: list[str],
) -> str:
    if urgency >= policy.urgent_threshold or heat >= policy.heat_urgent:
        rationale.append(
            f"срочность={urgency:.2f}, накал={heat:.2f} -> urgent"
        )
        return "urgent"
    if urgency >= policy.high_threshold or heat >= policy.heat_high or churn >= policy.churn_threshold:
        rationale.append(
            f"срочность={urgency:.2f}, накал={heat:.2f}, отток={churn:.2f} -> high"
        )
        return "high"
    if urgency < 0.2 and heat < 1.0:
        rationale.append("низкая срочность и низкий накал -> low")
        return "low"
    return "normal"


def _tone(heat: float, churn: float) -> str:
    if heat >= 3.0:
        return "извиняющийся"
    if heat >= 1.5 or churn >= 3.0:
        return "эмпатичный"
    return "нейтральный"


def _reply(team: str, priority: str, tone: str, escalate: bool) -> str:
    if tone == "извиняющийся":
        opening = "Приносим извинения за доставленные неудобства."
    elif tone == "эмпатичный":
        opening = "Понимаю, как это неприятно."
    else:
        opening = "Спасибо за обращение."
    if escalate:
        return (
            f"{opening} Передаю обращение старшему инженеру "
            f"(команда «{team}», приоритет: {priority})."
        )
    return f"{opening} Обращение принято в команду «{team}» (приоритет: {priority})."


def route_ticket(
    decisions: dict[str, Any],
    policy: RoutingPolicy | None = None,
) -> AgentAction:
    """Превращает решения по обращению в действие чат-агента.

    Ожидаемые вопросы (любой можно опустить): ``срочность``/``эскалация`` (noul),
    ``команда`` (choice), ``накал``/``удержание`` (score).
    """
    policy = policy or RoutingPolicy()
    rationale: list[str] = []

    team = _choice(decisions, "команда") or policy.default_team
    if team != policy.default_team:
        rationale.append(f"команда из решения: {team!r}")

    urgency = _noul(decisions, "срочность") or 0.0
    escalation = _noul(decisions, "эскалация") or 0.0
    heat = _score(decisions, "накал") or 0.0
    churn = _score(decisions, "удержание") or 0.0

    priority = _priority(urgency, heat, churn, policy, rationale)
    escalate = escalation >= policy.escalate_threshold or priority == "urgent"
    if escalate:
        rationale.append(
            f"эскалация={escalation:.2f} (порог {policy.escalate_threshold:.2f})"
        )
    tone = _tone(heat, churn)
    rationale.append(f"тон ответа: {tone}")

    return AgentAction(
        team=team,
        priority=priority,
        escalate=escalate,
        tone=tone,
        reply=_reply(team, priority, tone, escalate),
        rationale=tuple(rationale),
    )
