"""Минимальный UCI-клиент: движок как подпроцесс, обмен строками.

UCI — штатный машинный протокол Stockfish, поэтому пример не читает экран и не
нажимает клавиши: только stdin/stdout. Этого достаточно, чтобы играть партию,
просить несколько лучших ходов (`MultiPV`) и ограничивать силу соперника.
"""
from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Self

#: Мат в сантипешках: больше любой реальной оценки, меньше — любая потеря.
MATE_SCORE = 100_000

_MULTIPV = re.compile(r"\bmultipv (\d+)\b")
_SCORE_CP = re.compile(r"\bscore cp (-?\d+)\b")
_SCORE_MATE = re.compile(r"\bscore mate (-?\d+)\b")
_PV = re.compile(r"\bpv (.+)$")


@dataclass(frozen=True)
class Candidate:
    """Один ход из выдачи движка: оценка и главная линия."""

    move: str
    cp: int | None = None
    mate: int | None = None
    pv: tuple[str, ...] = ()

    @property
    def score(self) -> int:
        """Оценка в сантипешках; мат переводится в заведомо большую величину."""
        if self.mate is not None:
            sign = 1 if self.mate > 0 else -1
            return sign * (MATE_SCORE - abs(self.mate))
        return 0 if self.cp is None else self.cp

    def describe(self) -> str:
        """Короткая подпись для лога: `g1f3 +0.32` или `d8h4 #3`."""
        if self.mate is not None:
            return f"{self.move} #{self.mate}"
        return f"{self.move} {self.score / 100:+.2f}"


def parse_info(line: str) -> tuple[int, Candidate] | None:
    """Строка `info` -> (номер multipv, кандидат). Без `pv` строка не нужна."""
    if not line.startswith("info ") or " pv " not in line:
        return None
    pv_match = _PV.search(line)
    if pv_match is None:
        return None
    moves = tuple(pv_match.group(1).split())
    if not moves:
        return None
    index = _MULTIPV.search(line)
    cp = _SCORE_CP.search(line)
    mate = _SCORE_MATE.search(line)
    return (
        int(index.group(1)) if index else 1,
        Candidate(
            move=moves[0],
            cp=int(cp.group(1)) if cp else None,
            mate=int(mate.group(1)) if mate else None,
            pv=moves,
        ),
    )


class UciEngine:
    """UCI-движок с ограничением силы и мультивариантным анализом."""

    def __init__(
        self,
        path: str,
        *,
        threads: int = 1,
        hash_mb: int = 64,
        skill: int | None = None,
        multipv: int = 1,
        timeout: float = 60.0,
    ):
        self.path = path
        self.threads = threads
        self.hash_mb = hash_mb
        self.skill = skill
        self.multipv = multipv
        self.timeout = timeout
        self._proc: subprocess.Popen[str] | None = None
        # Читатель живёт в отдельном потоке: смешивать select() с readline()
        # нельзя — текстовая обёртка забирает данные в свой буфер, и fd молчит.
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self.name = "unknown"

    # --- жизненный цикл -----------------------------------------------------
    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        self._reader = threading.Thread(
            target=self._pump, args=(self._proc.stdout,), daemon=True
        )
        self._reader.start()
        self._send("uci")
        for line in self._wait_for("uciok"):
            if line.startswith("id name "):
                self.name = line[len("id name ") :]
        self.set("Threads", self.threads)
        self.set("Hash", self.hash_mb)
        self.set("MultiPV", self.multipv)
        if self.skill is not None:
            self.set("Skill Level", self.skill)
        self._send("isready")
        self._wait_for("readyok")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._send("quit")
            self._proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()
        finally:
            self._proc = None

    # --- обмен --------------------------------------------------------------
    def _send(self, command: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("engine is not running")
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def set(self, name: str, value: object) -> None:
        self._send(f"setoption name {name} value {value}")

    def _pump(self, stream) -> None:
        """Переносит stdout движка в очередь строк; None — конец потока."""
        try:
            for line in stream:
                self._lines.put(line.rstrip("\n"))
        finally:
            self._lines.put(None)

    def _readline(self, deadline: float) -> str | None:
        """Одна строка от движка или None, если вышли за дедлайн."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            item = self._lines.get(timeout=remaining)
        except queue.Empty:
            return None
        if item is None:
            raise RuntimeError("engine closed stdout unexpectedly")
        return item

    def _wait_for(self, token: str) -> list[str]:
        deadline = time.monotonic() + self.timeout
        seen: list[str] = []
        while True:
            line = self._readline(deadline)
            if line is None:
                raise TimeoutError(f"engine did not answer {token!r}")
            seen.append(line)
            if line.startswith(token):
                return seen

    def analyse(
        self, fen: str, *, movetime_ms: int, multipv: int | None = None
    ) -> list[Candidate]:
        """Топ ходов для позиции: `multipv` вариантов за `movetime_ms`."""
        if multipv is not None and multipv != self.multipv:
            self.set("MultiPV", multipv)
            self.multipv = multipv
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")
        deadline = time.monotonic() + self.timeout
        best: dict[int, Candidate] = {}
        while True:
            line = self._readline(deadline)
            if line is None:
                raise TimeoutError("engine did not return bestmove")
            if line.startswith("bestmove"):
                break
            parsed = parse_info(line)
            if parsed is not None:
                index, candidate = parsed
                # последняя (самая глубокая) строка по каждому multipv
                best[index] = candidate
        return [best[index] for index in sorted(best)]

    def best_move(self, fen: str, *, movetime_ms: int) -> str:
        """Топ-1 ход — соперник в демо-партии."""
        candidates = self.analyse(fen, movetime_ms=movetime_ms)
        if not candidates:
            raise RuntimeError("engine returned no candidates")
        return candidates[0].move
