"""Тесты выбора устройства и типа вычислений (DECISION_DEVICE/DECISION_DTYPE)."""
import pytest
import torch

from src import device as device_mod


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device_mod.resolve_device("auto") == "cuda"
    assert device_mod.resolve_device(None) == "cuda"
    assert device_mod.resolve_device("") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device_mod.resolve_device("auto") == "cpu"


def test_resolve_device_explicit_cpu_never_touches_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("cuda must not be queried")
    )
    assert device_mod.resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_index_preserved(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert device_mod.resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_cuda_unavailable_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        device_mod.resolve_device("cuda")


def test_resolve_device_cuda_index_out_of_range_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        device_mod.resolve_device("cuda:3")


def test_resolve_device_rejects_garbage():
    with pytest.raises(RuntimeError, match="not a valid device"):
        device_mod.resolve_device("not-a-device")


def test_resolve_dtype_defaults_and_values():
    assert device_mod.resolve_dtype(None) is torch.float32
    assert device_mod.resolve_dtype("") is torch.float32
    assert device_mod.resolve_dtype("float32") is torch.float32
    assert device_mod.resolve_dtype(" float16 ") is torch.float16
    assert device_mod.resolve_dtype("bfloat16") is torch.bfloat16


def test_resolve_dtype_rejects_unknown():
    with pytest.raises(RuntimeError, match="DECISION_DTYPE"):
        device_mod.resolve_dtype("int8")


def test_check_dtype_device_rejects_fp16_on_cpu():
    with pytest.raises(RuntimeError, match="float16"):
        device_mod.check_dtype_device(torch.float16, "cpu")


def test_check_dtype_device_allows_fp16_on_cuda():
    device_mod.check_dtype_device(torch.float16, "cuda")
    device_mod.check_dtype_device(torch.float32, "cpu")
    device_mod.check_dtype_device(torch.bfloat16, "cpu")
