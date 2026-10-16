"""Smoke-тест примеров: offline-примеры действительно запускаются.

Примеры импортируются и выполняются в процессе (без повторного старта
интерпретатора), поэтому тест быстрый. Сетевые примеры проверяются через
``--help``: они требуют сервера, но должны корректно собирать аргументы.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# (путь, аргументы) — примеры, полностью работающие без сервера и сети.
# Плейсхолдер ``{tmp}`` заменяется на tmp_path.
OFFLINE_CASES = [
    ("examples/quickstart/hello.py", []),
    ("examples/batch/inprocess_batch.py", ["--n", "3"]),
    ("examples/batch/csv_triage.py", ["--n", "3", "--output", "{tmp}/triage.csv"]),
    ("examples/cascade/hash_to_llm.py", []),
    ("examples/cascade/risk_aware.py", ["--budget", "0.3"]),
    ("examples/vision/vision_offline.py", []),
    ("examples/vision/make_sample_images.py", ["--n", "2", "--out", "{tmp}/img"]),
    ("examples/vision/image_single.py", ["--image", "{tmp}/img/img_00000.png"]),
    ("examples/vision/image_folder.py", ["--dir", "{tmp}/img2", "--show-first", "0"]),
    (
        "examples/vision/video_frames.py",
        ["--dir", "{tmp}/img2", "--clip", "{tmp}/clip.mp4"],
    ),
    ("examples/integrations/email_triage.py", []),
    ("examples/integrations/support_copilot.py", []),
    ("examples/integrations/moderation.py", []),
    ("examples/integrations/sentiment_router.py", []),
    ("examples/integrations/openai_tools.py", []),
    ("examples/integrations/telegram_bot.py", []),
    ("examples/integrations/discord_bot.py", []),
    ("examples/integrations/webhook_fastapi.py", []),
    ("examples/integrations/slack_app.py", []),
    ("examples/ops/option_transfer.py", ["--train-steps", "300", "--n-per-label", "10"]),
    ("examples/ops/local_heads.py", ["--dir", "{tmp}/heads"]),
    ("examples/ops/device_select.py", ["--n", "10"]),
]

# (путь, аргументы) — примеры, требующие сервера: проверяем разбор аргументов.
HELP_CASES = [
    ("examples/batch/jsonl_batch.py", ["--help"]),
    ("examples/protocol/systemone_client.py", ["--help"]),
    ("examples/cascade/client_fallback.py", ["--help"]),
    ("examples/ops/weight_pack.py", ["--help"]),
    ("examples/integrations/chat_agent.py", ["--help"]),
]


def _run_module(rel_path: str, argv: list[str]) -> None:
    """Импортирует пример и вызывает его ``main()`` с заданными аргументами."""
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    old_argv = sys.argv
    sys.argv = [str(path), *argv]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def _ids(cases: list[tuple[str, list[str]]]) -> list[str]:
    return [Path(rel).stem for rel, _ in cases]


@pytest.mark.parametrize("rel_path,argv", OFFLINE_CASES, ids=_ids(OFFLINE_CASES))
def test_offline_example(rel_path: str, argv: list[str], tmp_path: Path) -> None:
    resolved = [arg.format(tmp=tmp_path) for arg in argv]
    _run_module(rel_path, resolved)


@pytest.mark.parametrize("rel_path,argv", HELP_CASES, ids=_ids(HELP_CASES))
def test_network_example_help(rel_path: str, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_module(rel_path, argv)
    assert excinfo.value.code == 0
