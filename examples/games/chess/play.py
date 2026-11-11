"""Партия в шахматы: движок предлагает ходы, наша голова решает, когда думать.

Разделение ролей намеренное:

* Stockfish даёт 3 лучших хода быстрым просчётом — это «генератор вариантов»;
* голова `need_deep` (обучена на наших данных) отвечает, надёжен ли быстрый
  ответ, и только тогда мы платим за глубокий просчёт;
* ход — типизированное решение, а не текст: `Choice` из кандидатов, где выбор
  делает результат головы.

Запуск (нужен Stockfish и обученный пак, см. README примера):

    python -m examples.games.chess.play --engine .tools/stockfish \
        --pack checkpoints/packs/chess --skill 0

Демо-режим для записи экрана печатает кадры с паузами и цветом:

    python -m examples.games.chess.play --engine .tools/stockfish \
        --pack checkpoints/packs/chess --demo --pgn data/chess/game.pgn
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import chess.pgn

from eval.datasets.chess import CHESS_SPECS
from examples.games.chess.engine import UciEngine
from examples.games.chess.state import describe_move, render, text_state

CLEAR = "\x1b[2J\x1b[H"
DIM = "\x1b[90m"
BOLD = "\x1b[1m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"


def load_service(pack: Path, dim: int | None):
    """Сервис решений на паке голов; без пака — явная ошибка, а не догадки."""
    from api.server.service import DecisionService, HashStateEncoder

    metadata = pack / "metadata.json"
    if not metadata.is_file():
        raise SystemExit(
            f"нет пака голов в {pack}: сначала обучите его\n"
            "  python -m examples.games.chess.dataset --engine <stockfish>\n"
            "  python -m training.recipe.train_heads --specs chess "
            f"--dataset data/chess --encoder hash --dim 256 --out-dir {pack}"
        )
    feature_dim = dim or json.loads(metadata.read_text(encoding="utf-8"))["feature_dim"]
    return DecisionService(
        encoder=HashStateEncoder(feature_dim),
        dim=feature_dim,
        checkpoint_dir=pack,
        untrained_policy="refuse",
    )


def our_move(
    board: chess.Board,
    service,
    analyst: UciEngine,
    *,
    shallow_ms: int,
    deep_ms: int,
    threshold: float,
) -> tuple[chess.Move, dict]:
    """Один ход: кандидаты -> решение головы -> (при необходимости) глубокий поиск."""
    fen = board.fen()
    candidates = analyst.analyse(fen, movetime_ms=shallow_ms, multipv=3)
    if not candidates:
        raise RuntimeError("движок не вернул кандидатов")
    state = text_state(board, candidates)
    decisions = service.decide(state, CHESS_SPECS)
    # in-process сервис отдаёт типизированные результаты, а не JSON
    need_deep = decisions["need_deep"]
    escalated = float(need_deep.noul) >= threshold
    chosen = candidates[0]
    if escalated:
        deep = analyst.analyse(fen, movetime_ms=deep_ms, multipv=1)
        chosen = deep[0]
    return (
        chess.Move.from_uci(chosen.move),
        {
            "state": state,
            "candidates": candidates,
            "need_deep": float(need_deep.noul),
            "confidence": float(need_deep.confidence),
            "escalated": escalated,
            "chosen": chosen.describe(),
        },
    )


def frame(
    board: chess.Board,
    *,
    number: int,
    side: str,
    info: dict | None,
    last_move: chess.Move | None,
    color: bool,
) -> str:
    """Один кадр: доска, кандидаты, вердикт головы."""
    lines = [f"{BOLD}decision-cascade · шахматы{RESET}"]
    lines.append(render(board, last_move=last_move, color=color))
    if info is not None:
        options = " / ".join(c.describe() for c in info["candidates"])
        lines.append(f"{DIM}ход {number} ({side}) · кандидаты: {options}{RESET}")
        verdict = (
            f"{YELLOW}need_deep={info['need_deep']:.2f} → глубокий просчёт{RESET}"
            if info["escalated"]
            else f"{GREEN}need_deep={info['need_deep']:.2f} → быстрый ответ{RESET}"
        )
        lines.append(verdict + f"{DIM} · выбран {info['chosen']}{RESET}")
    return "\n".join(lines)


def play(
    *,
    engine_path: str,
    pack: Path,
    skill: int = 0,
    max_plies: int = 120,
    shallow_ms: int = 60,
    deep_ms: int = 260,
    opponent_ms: int = 60,
    threshold: float = 0.5,
    color: bool = False,
    delay: float = 0.0,
    quiet: bool = False,
    show_fen: bool = True,
    show_stats: bool = False,
) -> dict:
    service = load_service(pack, None)
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    stats = {"plies": 0, "escalations": 0, "our_ms": 0.0, "moves": []}

    with (
        UciEngine(engine_path, multipv=3, hash_mb=256) as analyst,
        UciEngine(engine_path, skill=skill, hash_mb=64) as opponent,
    ):
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
            if board.turn == chess.WHITE:
                started = time.perf_counter()
                move, info = our_move(
                    board,
                    service,
                    analyst,
                    shallow_ms=shallow_ms,
                    deep_ms=deep_ms,
                    threshold=threshold,
                )
                stats["our_ms"] += (time.perf_counter() - started) * 1000
                stats["escalations"] += int(info["escalated"])
                stats["moves"].append(
                    {
                        "ply": len(board.move_stack) + 1,
                        "need_deep": info["need_deep"],
                        "escalated": info["escalated"],
                    }
                )
                side = "наш ход"
            else:
                move = chess.Move.from_uci(
                    opponent.best_move(board.fen(), movetime_ms=opponent_ms)
                )
                info = None
                side = "соперник"

            if move not in board.legal_moves:
                raise RuntimeError(f"нелегальный ход от движка: {move.uci()}")
            san = describe_move(board, move)
            board.push(move)
            node = node.add_variation(move)
            stats["plies"] = len(board.move_stack)

            if not quiet:
                print(CLEAR if delay else "", end="")
                print(
                    frame(
                        board,
                        number=len(board.move_stack),
                        side=side,
                        info=info,
                        last_move=move,
                        color=color,
                    )
                )
                if show_fen:
                    print(f"{DIM}{board.fen()}{RESET}")
                ply = len(board.move_stack)
                number = (ply + 1) // 2
                # чёрные пишутся как «4... d6», иначе лог читается как шахматная нотация
                label = f"{number}. {san}" if ply % 2 else f"{number}... {san}"
                print(f"{BOLD}{label}{RESET}")
                if show_stats:
                    played = len(board.move_stack) // 2
                    print(
                        f"{DIM}эскалаций: {stats['escalations']} из {played} "
                        f"({stats['escalations'] / max(1, played):.0%}){RESET}"
                    )
                if delay:
                    time.sleep(delay)

    game.headers["Event"] = "decision-cascade demo"
    game.headers["White"] = "decision-cascade (candidates + need_deep head)"
    game.headers["Black"] = f"Stockfish skill {skill}"
    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    game.headers["Result"] = result
    stats["result"] = result
    stats["escalation_rate"] = (
        stats["escalations"] / (stats["plies"] // 2) if stats["plies"] else 0.0
    )
    stats["avg_our_ms"] = stats["our_ms"] / max(1, stats["plies"] // 2)
    stats["game"] = game
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="путь к бинарнику Stockfish")
    parser.add_argument("--pack", type=Path, default=Path("checkpoints/packs/chess"))
    parser.add_argument("--skill", type=int, default=0, help="сила соперника 0-20")
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--shallow-ms", type=int, default=60)
    parser.add_argument("--deep-ms", type=int, default=260)
    parser.add_argument("--opponent-ms", type=int, default=60)
    # порог выбран по кривой из evaluate.py: ~19% эскалаций закрывают половину
    # разрыва до «всегда глубоко» (см. README примера)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--pgn", type=Path, default=None)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="цвет, паузы между кадрами и очистка экрана — для записи",
    )
    parser.add_argument("--delay", type=float, default=0.55)
    args = parser.parse_args(argv)

    stats = play(
        engine_path=args.engine,
        pack=args.pack,
        skill=args.skill,
        max_plies=args.max_plies,
        shallow_ms=args.shallow_ms,
        deep_ms=args.deep_ms,
        opponent_ms=args.opponent_ms,
        threshold=args.threshold,
        color=args.demo,
        delay=args.delay if args.demo else 0.0,
        # FEN в кадре — отладочный шум, в демо он только занимает строку
        show_fen=not args.demo,
        show_stats=args.demo,
    )

    print(
        f"\n{BOLD}Итог: {stats['result']}{RESET} · ходов {stats['plies']} · "
        f"эскалаций {stats['escalations']} ({stats['escalation_rate']:.0%}) · "
        f"среднее на наш ход {stats['avg_our_ms']:.0f} мс"
    )
    if args.pgn is not None:
        args.pgn.parent.mkdir(parents=True, exist_ok=True)
        with args.pgn.open("w", encoding="utf-8") as handle:
            print(stats["game"], file=handle)
        print(f"{DIM}PGN -> {args.pgn}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
