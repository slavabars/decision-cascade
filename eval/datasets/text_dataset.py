"""Генератор синтетических обращений в поддержку с типизированными метками.

Состояние строится из латентного сценария (команда, причина, срочность, накал,
эскалация, риск оттока). Текст содержит ключевые слова, коррелирующие с меткой,
поэтому головы могут выучить соответствие, а бенчмарк — измерить качество.
"""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any

from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec

QUESTIONS: dict[str, dict[str, Any]] = {
    "срочность": {
        "type": "noul",
        "instructions": "Клиент сигнализирует о срочности и потере выручки",
    },
    "команда": {
        "type": "choice",
        "instructions": "Какая команда должна взять обращение",
        "options": ["биллинг", "интеграция", "общая поддержка", "продажи"],
    },
    "причина": {
        "type": "choice",
        "instructions": "Наиболее вероятная причина сбоя",
        "options": [
            "проблема с ключами API",
            "региональные ограничения",
            "квота аккаунта",
            "конфликт плагина",
        ],
    },
    "накал": {
        "type": "score",
        "instructions": "Насколько клиент раздражён",
        "levels": ["спокоен", "слегка недоволен", "умеренно", "сильно", "на грани"],
    },
    "эскалация": {
        "type": "noul",
        "instructions": "Обращение стоит эскалировать на senior-инженера",
    },
    "удержание": {
        "type": "score",
        "instructions": "Оценка риска оттока клиента",
        "levels": [
            "точно останется",
            "вероятно останется",
            "неопределённо",
            "вероятно уйдёт",
            "уйдёт наверняка",
        ],
    },
}


def spec_from_api(data: dict[str, Any]) -> QuestionSpec:
    kind = data["type"]
    criteria = data.get("criteria")
    if kind == "noul":
        return NoulSpec(
            instructions=data["instructions"],
            criteria=tuple(criteria) if criteria else None,
        )
    if kind == "choice":
        return ChoiceSpec(
            instructions=data["instructions"],
            options=tuple(data["options"]),
            criteria=dict(criteria) if criteria else None,
        )
    if kind == "score":
        return ScoreSpec(
            instructions=data["instructions"],
            levels=tuple(data["levels"]),
            criteria=tuple(criteria) if criteria else None,
        )
    raise ValueError(f"unsupported question type: {kind!r}")


SPECS: dict[str, QuestionSpec] = {
    name: spec_from_api(data) for name, data in QUESTIONS.items()
}

OPENERS = {
    "биллинг": [
        "Здравствуйте, у нас проблема с оплатой и списаниями.",
        "Добрый день, не приходят выплаты и счёт по Stripe расходится.",
        "Помогите, списание прошло, но платёж не зачислен.",
    ],
    "интеграция": [
        "Здравствуйте, не работает интеграция по API.",
        "Добрый день, наш вебхук и ключи перестали приниматься.",
        "Помогите подключить модуль, ключ не проходит.",
    ],
    "продажи": [
        "Здравствуйте, интересует апгрейд тарифа.",
        "Добрый день, нужен план и лицензия для расширения.",
        "Подскажите стоимость договора и смены тарифа.",
    ],
    "общая поддержка": [
        "Здравствуйте, подскажите, как настроить интерфейс.",
        "Добрый день, не понимаю, где найти нужную настройку.",
        "Помогите разобраться с гайдом по интерфейсу.",
    ],
}

CAUSE_PHRASES = {
    "проблема с ключами API": "ключи перестали приниматься, выдаёт 'invalid key'",
    "региональные ограничения": "сервис блокирует наш регион и не даёт работать",
    "квота аккаунта": "закончилась квота запросов на аккаунте",
    "конфликт плагина": "после обновления плагина всё сломалось",
}

URGENCY_PHRASES = {
    1: [
        "Это срочно, уже третьи сутки теряем выручку.",
        "Business-critical: простой стоит нам денег каждый час.",
    ],
    0: [
        "Особой спешки нет, вопрос плановый.",
        "Можно разобраться в рабочем порядке.",
    ],
}

IRRITATION_PHRASES = [
    "Говорю спокойно, без претензий.",
    "Немного раздражает, что приходится писать.",
    "Честно говоря, уже надоело ждать.",
    "Мы в ярости от такого сервиса!",
    "Это возмутительно, готовы судиться!",
]

ESCALATION_PHRASES = {
    1: "Прошу подключить старшего инженера, сами не справимся.",
    0: "Думаю, вопрос решается на первой линии.",
}

CHURN_PHRASES = {
    4: "Иначе уйдём наверняка к другому вендору.",
    3: "Если не решите, вероятно уйдём к конкурентам.",
    2: "Пока не определились, останемся мы или нет.",
    1: "Скорее всего останемся с вами.",
    0: "Точно останемся, просто хотим починить.",
}


def generate_examples(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = Random(seed)
    teams = list(OPENERS)
    causes = list(CAUSE_PHRASES)
    examples: list[dict[str, Any]] = []
    for i in range(n):
        team = rng.choice(teams)
        cause = rng.choice(causes)
        urgency = rng.randint(0, 1)
        irritation = rng.randint(0, 4)
        escalation = 1 if urgency and irritation >= 2 else rng.randint(0, 1)
        churn = min(4, irritation + (1 if urgency else 0) + rng.randint(0, 1))
        state = " ".join(
            [
                rng.choice(OPENERS[team]),
                f"Похоже, {CAUSE_PHRASES[cause]}.",
                rng.choice(URGENCY_PHRASES[urgency]),
                IRRITATION_PHRASES[irritation],
                ESCALATION_PHRASES[escalation],
                CHURN_PHRASES[churn],
            ]
        )
        examples.append(
            {
                "id": f"ticket-{i:05d}",
                "state": state,
                "labels": {
                    "срочность": urgency,
                    "команда": team,
                    "причина": cause,
                    "накал": irritation,
                    "эскалация": escalation,
                    "удержание": churn,
                },
            }
        )
    return examples


def split_examples(
    examples: list[dict[str, Any]],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Детерминированно делит примеры на train/val/test без пересечений."""
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("fractions must be in [0.0, 1.0)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")
    shuffled = list(examples)
    Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = round(n * test_fraction)
    n_val = round(n * val_fraction)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return {"train": train, "val": val, "test": test}


def write_jsonl(path: str | Path, examples: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
    return path
