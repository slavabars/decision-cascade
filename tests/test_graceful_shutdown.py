"""Аккуратное завершение: SIGTERM не обрывает уже принятый запрос.

Проверяем контракт, на который опирается systemd-юнит
(`KillSignal=SIGTERM`, `--timeout-graceful-shutdown 30`): uvicorn перестаёт
принимать новое, но доводит запрос, который уже выполняется.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signals only")

APP = '''
import asyncio


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    await asyncio.sleep(1.0)
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})
'''


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"uvicorn exited early: {proc.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError("uvicorn did not start listening")


def test_sigterm_waits_for_inflight_request(tmp_path: Path):
    (tmp_path / "slow_app.py").write_text(APP, encoding="utf-8")
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "slow_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--timeout-graceful-shutdown",
            "30",
            "--log-level",
            "warning",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port, proc)
        # Запрос уже в обработке (endpoint спит 1 с), когда приходит SIGTERM.
        with (
            httpx.Client(timeout=10.0) as client,
            client.stream("GET", f"http://127.0.0.1:{port}/") as response,
        ):
            time.sleep(0.2)
            proc.send_signal(signal.SIGTERM)
            body = b"".join(response.iter_bytes())
            assert response.status_code == 200
            assert body == b"ok"
        # uvicorn завершается с тем же сигналом, который получил (SIGTERM),
        # но только после того, как довёл текущий запрос.
        assert proc.wait(timeout=10) in {0, -int(signal.SIGTERM)}
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
