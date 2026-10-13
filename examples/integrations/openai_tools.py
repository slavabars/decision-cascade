"""Решения как инструмент (function calling) для LLM-агента.

Строит JSON-схему инструмента ``analyze_ticket`` и показывает полный цикл:
«модель» выбирает инструмент и аргументы -> мы выполняем его in-process ->
возвращаем результат в формате tool-сообщения. Годится как точка входа для
OpenAI/Anthropic/LangChain: подставьте свою схему и обработчик.

Запуск:

    .venv/bin/python examples/integrations/openai_tools.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples._common import build_service, decision_to_dict, ticket_specs

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_ticket",
        "description": "Разобрать обращение в поддержку: команда, причина, "
        "срочность, эскалация, накал и риск оттока.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "Текст обращения клиента целиком.",
                }
            },
            "required": ["state"],
        },
    },
}

# Так выглядел бы ответ модели, если бы она решила вызвать инструмент.
MODEL_TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {
        "name": "analyze_ticket",
        "arguments": json.dumps(
            {
                "state": "Не приходит выплата Stripe третьи сутки, теряем выручку!"
            },
            ensure_ascii=False,
        ),
    },
}


def analyze_ticket(state: str) -> dict:
    service = build_service()
    decisions = service.decide(state, ticket_specs())
    return {
        name: decision_to_dict(decision) for name, decision in decisions.items()
    }


def execute_tool_call(call: dict) -> str:
    name = call["function"]["name"]
    if name != "analyze_ticket":
        raise ValueError(f"unknown tool: {name}")
    arguments = json.loads(call["function"]["arguments"])
    return json.dumps(analyze_ticket(arguments["state"]), ensure_ascii=False)


def main() -> None:
    print("Схема инструмента:")
    print(json.dumps(TOOL_SCHEMA, ensure_ascii=False, indent=2))
    print("\nМодель вызвала инструмент:")
    print(json.dumps(MODEL_TOOL_CALL, ensure_ascii=False, indent=2))
    result = execute_tool_call(MODEL_TOOL_CALL)
    print("\nРезультат (tool-сообщение для модели):")
    print(result)


if __name__ == "__main__":
    main()
