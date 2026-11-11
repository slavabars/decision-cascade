"""Шахматный пример: протокол UCI, признаки позиции и логика каскада.

Тесты не запускают Stockfish: парсинг UCI-строк и решение «нужен ли глубокий
просчёт» проверяются на заглушках, поэтому набор зелёный без движка и без сети.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.datasets.chess import CHESS_QUESTIONS, CHESS_SPECS, load_splits
from examples.games.chess import dataset, evaluate
from examples.games.chess.engine import Candidate, parse_info

pytest.importorskip("chess")
import chess

from examples.games.chess import play, state


# --- UCI -------------------------------------------------------------------
def test_parse_info_with_multipv():
    index, candidate = parse_info(
        "info depth 18 multipv 2 score cp -34 nodes 1 pv g8f6 f3e5"
    )
    assert index == 2
    assert candidate.move == "g8f6"
    assert candidate.score == -34
    assert candidate.pv[:2] == ("g8f6", "f3e5")


def test_parse_info_defaults_to_first_line_and_skips_noise():
    index, candidate = parse_info("info depth 12 score cp 25 pv e2e4 e7e5")
    assert index == 1
    assert candidate.move == "e2e4"
    assert parse_info("info depth 12 nodes 100") is None
    assert parse_info("Stockfish 19") is None


def test_mate_is_bigger_than_any_pawn_score():
    _, mate = parse_info("info depth 20 multipv 1 score mate 3 pv d8h4 g7g6")
    _, cp = parse_info("info depth 20 multipv 1 score cp 900 pv d8h4")
    assert mate.score > cp.score
    assert mate.describe() == "d8h4 #3"
    assert cp.describe() == "d8h4 +9.00"


# --- метка и признаки ------------------------------------------------------
def test_is_unreliable_flags_move_change_and_big_score_drift():
    shallow = [Candidate("e2e4", cp=20)]
    assert dataset.is_unreliable(shallow, [Candidate("d2d4", cp=25)])
    assert dataset.is_unreliable(shallow, [Candidate("e2e4", cp=120)])
    assert not dataset.is_unreliable(shallow, [Candidate("e2e4", cp=40)])


def test_text_state_has_no_deep_search_information():
    board = chess.Board()
    candidates = [Candidate("e2e4", cp=25), Candidate("d2d4", cp=20)]
    text = state.text_state(board, candidates)
    assert text.startswith("chess ")
    assert "phase=opening" in text
    assert "spread=1-15" in text
    # признаки обязаны быть известны до выбора хода: ничего из «глубокого» ответа
    for forbidden in ("need_deep", "delta_cp", "deep="):
        assert forbidden not in text


def test_material_is_relative_to_side_to_move():
    board = chess.Board()
    board.remove_piece_at(chess.A8)  # чёрная ладья
    assert state.material(board) == 5  # ходят белые
    board.turn = chess.BLACK
    assert state.material(board) == -5


def test_render_marks_last_move_and_coordinates():
    board = chess.Board()
    board.push_uci("e2e4")
    plain = state.render(board, last_move=chess.Move.from_uci("e2e4"))
    assert "[P]" in plain
    assert " a " in plain and " h " in plain
    colored = state.render(board, last_move=chess.Move.from_uci("e2e4"), color=True)
    assert "\x1b[7m" in colored


# --- логика каскада --------------------------------------------------------
def _service(prob: float):
    return SimpleNamespace(
        decide=lambda _state, _specs: {
            "need_deep": SimpleNamespace(noul=prob, confidence=max(prob, 1 - prob))
        }
    )


class _Analyst:
    """Заглушка движка: быстрый ответ — e2e4, глубокий — d2d4."""

    def __init__(self):
        self.calls: list[int] = []

    def analyse(self, _fen, *, movetime_ms, multipv=None):
        self.calls.append(movetime_ms)
        if movetime_ms >= 200:
            return [Candidate("d2d4", cp=30)]
        return [Candidate("e2e4", cp=25), Candidate("d2d4", cp=22)]


def test_our_move_uses_fast_answer_when_head_is_confident():
    analyst = _Analyst()
    move, info = play.our_move(
        chess.Board(),
        _service(0.1),
        analyst,
        shallow_ms=50,
        deep_ms=250,
        threshold=0.5,
    )
    assert move.uci() == "e2e4"
    assert info["escalated"] is False
    assert analyst.calls == [50]


def test_our_move_escalates_to_deep_search():
    analyst = _Analyst()
    move, info = play.our_move(
        chess.Board(),
        _service(0.9),
        analyst,
        shallow_ms=50,
        deep_ms=250,
        threshold=0.5,
    )
    assert move.uci() == "d2d4"
    assert info["escalated"] is True
    assert analyst.calls == [50, 250]


# --- метрики оценки --------------------------------------------------------
def test_auc_known_values():
    import numpy as np

    perfect = evaluate.auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert perfect == pytest.approx(1.0)
    inverted = evaluate.auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0]))
    assert inverted == pytest.approx(0.0)
    # все скоры равны — AUC 0.5, а не деление на ноль
    assert evaluate.auc(np.array([0.5, 0.5]), np.array([0, 1])) == pytest.approx(0.5)


def test_quality_curve_trades_quality_for_cost():
    import numpy as np

    examples = [
        {"shallow": "e2e4 +0.20", "deep": "e2e4 +0.25"},
        {"shallow": "g1f3 +0.10", "deep": "d2d4 +0.60"},
    ]
    probs = np.array([0.1, 0.8])
    rows = evaluate.quality_curve(probs, examples, shallow_cost=60, deep_cost=200)
    by_threshold = {row["threshold"]: row for row in rows}
    # низкий порог — эскалируем всё, качество максимальное
    assert by_threshold[0.1]["quality"] == pytest.approx(1.0)
    assert by_threshold[0.1]["escalation_rate"] == pytest.approx(1.0)
    assert by_threshold[0.1]["avg_ms"] == pytest.approx(260)
    # высокий порог — не эскалируем ничего, качество падает вдвое
    assert by_threshold[0.9]["quality"] == pytest.approx(0.5)
    assert by_threshold[0.9]["avg_ms"] == pytest.approx(60)
    assert evaluate.agree_share(examples) == pytest.approx(0.5)


# --- данные ----------------------------------------------------------------
def test_load_splits_reads_directory_and_reports_missing(tmp_path: Path):
    for name in ("train", "val", "test"):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps({"state": "chess", "labels": {"need_deep": 0}}) + "\n",
            encoding="utf-8",
        )
    splits = load_splits(tmp_path)
    assert set(splits) == {"train", "val", "test"}
    assert splits["train"][0]["labels"]["need_deep"] == 0

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="нет сплитов"):
        load_splits(empty)


def test_load_splits_splits_single_file(tmp_path: Path):
    path = tmp_path / "all.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"state": f"chess i={i}", "labels": {"need_deep": i % 2}})
            for i in range(10)
        ),
        encoding="utf-8",
    )
    splits = load_splits(path)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 10
    assert len(splits["test"]) == 2


def test_chess_question_contract():
    assert set(CHESS_SPECS) == {"need_deep"}
    assert CHESS_QUESTIONS["need_deep"]["type"] == "noul"
