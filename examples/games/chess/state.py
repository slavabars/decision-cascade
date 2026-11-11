"""Текстовое состояние позиции и рендер доски.

Состояние — это то, что видит голова решения: дискретные признаки, которые
известны **до** выбора хода (материал, фаза, разброс оценок кандидатов). Здесь
нет ни одного признака, подсмотренного в глубоком поиске, — иначе метка
`need_deep` протекала бы в признаки.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:  # pragma: no cover - только для типов
    from examples.games.chess.engine import Candidate

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

#: Материал, который считается «фигурным» — по нему определяется фаза.
_NON_PAWN = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def material(board: chess.Board) -> int:
    """Материальный перевес с точки зрения стороны, которая ходит."""
    white = sum(
        len(board.pieces(piece, chess.WHITE)) * value
        for piece, value in PIECE_VALUE.items()
    )
    black = sum(
        len(board.pieces(piece, chess.BLACK)) * value
        for piece, value in PIECE_VALUE.items()
    )
    return (white - black) if board.turn == chess.WHITE else (black - white)


def non_pawn_material(board: chess.Board) -> int:
    return sum(
        len(board.pieces(piece, color)) * PIECE_VALUE[piece]
        for piece in _NON_PAWN
        for color in (chess.WHITE, chess.BLACK)
    )


def _bucket(value: int, edges: Sequence[int], labels: Sequence[str]) -> str:
    for edge, label in zip(edges, labels):
        if value <= edge:
            return label
    return labels[-1]


def phase(board: chess.Board) -> str:
    pieces = len(board.piece_map())
    if non_pawn_material(board) <= 12 or pieces <= 12:
        return "endgame"
    if board.fullmove_number <= 10 and pieces >= 28:
        return "opening"
    return "middlegame"


def material_label(board: chess.Board) -> str:
    value = material(board)
    return _bucket(value, (-3, -1, 1, 3), ("-3", "-2..-1", "even", "+1..2", "+3"))


def _move_facts(board: chess.Board) -> tuple[int, int, int]:
    """(сколько ходов, сколько взятий, сколько даёт шах)."""
    legal = 0
    captures = 0
    checks = 0
    for move in board.legal_moves:
        legal += 1
        if board.is_capture(move):
            captures += 1
        if board.gives_check(move):
            checks += 1
    return legal, captures, checks


def text_state(board: chess.Board, candidates: Sequence[Candidate]) -> str:
    """Компактное состояние: токены `ключ=значение` через пробел."""
    legal, captures, checks = _move_facts(board)
    scores = [candidate.score for candidate in candidates] or [0]
    top = scores[0]
    spread12 = top - scores[1] if len(scores) > 1 else 0
    spread13 = top - scores[2] if len(scores) > 2 else spread12
    best = candidates[0] if candidates else None
    best_move = chess.Move.from_uci(best.move) if best is not None else None
    tokens = [
        "chess",
        f"phase={phase(board)}",
        f"material={material_label(board)}",
        f"legal={_bucket(legal, (9, 19, 29, 39), ('0-9', '10-19', '20-29', '30-39', '40+'))}",
        f"check={'yes' if board.is_check() else 'no'}",
        f"captures={_bucket(captures, (0, 1, 3), ('0', '1', '2-3', '4+'))}",
        f"checks={_bucket(checks, (0, 1, 3), ('0', '1', '2-3', '4+'))}",
        f"spread={_bucket(spread12, (0, 15, 50, 150), ('0', '1-15', '16-50', '51-150', '150+'))}",
        f"spread3={_bucket(spread13, (20, 60, 150), ('0-20', '21-60', '61-150', '150+'))}",
        f"top1={_bucket(top, (-200, -50, 50, 200), ('losing', 'worse', 'even', 'better', 'winning'))}",
        f"best_capture={'yes' if best_move is not None and board.is_capture(best_move) else 'no'}",
        f"best_check={'yes' if best_move is not None and board.gives_check(best_move) else 'no'}",
    ]
    return " ".join(tokens)


_FILES = "abcdefgh"

#: Цвета: белые — яркие, чёрные — приглушённые, последний ход — инверсия.
_WHITE = "\x1b[97m"
_BLACK = "\x1b[90m"
_HILITE = "\x1b[7m"
_RESET = "\x1b[0m"


def render(
    board: chess.Board,
    *,
    last_move: chess.Move | None = None,
    color: bool = False,
) -> str:
    """Доска с координатами; последний ход подсвечен (цветом или скобками)."""
    rows = []
    for rank in range(8, 0, -1):
        cells = []
        for file in range(8):
            square = chess.square(file, rank - 1)
            piece = board.piece_at(square)
            symbol = piece.symbol() if piece else "."
            marked = last_move is not None and square in (
                last_move.from_square,
                last_move.to_square,
            )
            if not color:
                cells.append(f"[{symbol}]" if marked else f" {symbol} ")
                continue
            tint = ""
            if piece is not None:
                tint = _WHITE if piece.color == chess.WHITE else _BLACK
            prefix = _HILITE if marked else ""
            cells.append(f"{prefix}{tint} {symbol} {_RESET}")
        rows.append(f"{rank} " + "".join(cells))
    rows.append("   " + "".join(f" {letter} " for letter in _FILES))
    return "\n".join(rows)


def describe_move(board: chess.Board, move: chess.Move) -> str:
    """SAN-нотация хода — для читаемого лога партии."""
    try:
        return board.san(move)
    except ValueError:
        return move.uci()
