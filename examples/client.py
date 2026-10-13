"""Пример на русском: анализ обращения в техподдержку.

Показывает типизированные решения по шести вопросам разного типа
(noul / choice / score), получаемые одним запросом к Decision API.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.sdk.client import DecisionClient

STATE = (
    "Добрый день. Уже третьи сутки не могу подключить Stripe к нашему "
    "интернет-магазину. При попытке оплаты клиенты видят ошибку «400: "
    "Invalid account». Я пробовал пересоздать ключи, переустанавливал "
    "платёжный модуль — ничего не помогает. За это время мы потеряли "
    "порядка 15 заказов, оборот упал примерно на 40%. Учётка и карта — "
    "активные, но в дашборде тоже нет выплат за последнюю неделю. "
    "Если к пятнице не разберёмся, перейдём на другую платёжную систему."
)

QUESTIONS = {
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
        "options": ["проблема с ключами API", "региональные ограничения", "квота аккаунта", "конфликт плагина"],
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
        "levels": ["точно останется", "вероятно останется", "неопределённо", "вероятно уйдёт", "уйдёт наверняка"],
    },
}

TYPE_LABEL = {"noul": "Бинарный", "choice": "Выбор", "score": "Шкала"}


def print_decision(name: str, decision: dict) -> None:
    print(f"\n— {name} —  [{TYPE_LABEL[decision['type']]}]")
    if decision["type"] == "noul":
        print(f"    значение: {decision['noul']:.3f}   уверенность: {decision['confidence']:.3f}")
    elif decision["type"] == "choice":
        print(f"    выбор: '{decision['choice']}'   уверенность: {decision['confidence']:.3f}")
        for opt, prob in sorted(decision["probabilities"].items(), key=lambda kv: -kv[1]):
            print(f"      {opt:<22} {prob:.3f}")
    else:
        levels = decision["legend"]
        best = max(range(len(levels)), key=lambda i: decision["probabilities"][levels[i]])
        print(f"    балл: {decision['score']:.3f}  → «{levels[best]}»")
        for lvl, prob in sorted(decision["probabilities"].items(), key=lambda kv: -kv[1]):
            print(f"      {lvl:<22} {prob:.3f}")


def main() -> None:
    print("=== Анализ обращения в техподдержку (System One Decision Model) ===\n")
    print(f"Исходное обращение:\n    {STATE}\n")
    print(f"Запрос по {len(QUESTIONS)} вопросам к API …\n")

    with DecisionClient() as client:
        result: dict = client.decisions(STATE, QUESTIONS)

    print("-" * 52)
    for name, decision in result["decisions"].items():
        print_decision(name, decision)
    print("-" * 52)
    print(f"\nСхема: {result['schema_version']}")
    print("Интерпретация: показатели выше и вероятности по каждой опции — "
          "это решения модели по итогам анализа состояния.")


if __name__ == "__main__":
    main()