"""GPU-тесты: реальный forward на CUDA и совпадение с CPU.

Модель — крошечный случайный GPT2, собранный локально (без загрузки весов из
сети), поэтому тесты быстрые и воспроизводимые. Помечены ``gpu`` и при
отсутствии CUDA пропускаются автоматически.
"""
import copy

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


class _IdsTokenizer:
    """Детерминированный токенизатор: вторая строка батча имеет padding."""

    def __call__(self, states, **kwargs):
        n = len(states)
        ids = torch.ones(n, 4, dtype=torch.long)
        mask = torch.ones(n, 4)
        if n >= 2:
            mask[1, 2:] = 0.0  # padding-позиции не должны влиять на pooling
        return {"input_ids": ids, "attention_mask": mask}


def _tiny_gpt2():
    from transformers import GPT2Config, GPT2Model

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=64,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=1,
    )
    return GPT2Model(config).eval()


def test_resolve_device_cuda_real():
    from src.device import resolve_device

    assert resolve_device("cuda") == "cuda"


def test_cuda_tensor_roundtrip():
    tensor = torch.ones(4, device="cuda")
    assert tensor.sum().item() == 4.0
    assert tensor.to("cpu").device.type == "cpu"


def test_gemma_encode_on_cuda_matches_cpu():
    from src.encoder.gemma import GemmaStateEncoder

    cpu_model = _tiny_gpt2()
    gpu_model = copy.deepcopy(cpu_model).to("cuda")

    cpu_enc = GemmaStateEncoder(feature_dim=8, max_length=None, device="cpu")
    cpu_enc._model, cpu_enc._tokenizer = cpu_model, _IdsTokenizer()
    gpu_enc = GemmaStateEncoder(feature_dim=8, max_length=None, device="cuda")
    gpu_enc._model, gpu_enc._tokenizer = gpu_model, _IdsTokenizer()

    out_cpu = cpu_enc.encode(["first state", "second state"])
    out_gpu = gpu_enc.encode(["first state", "second state"])

    assert out_gpu.device.type == "cpu"  # наружу всегда CPU-тензор
    assert tuple(out_cpu.shape) == (2, 8)  # проекция 16 -> 8
    assert torch.allclose(out_cpu, out_gpu, atol=1e-4, rtol=1e-3)


def test_gemma_encode_fp16_on_cuda_runs():
    from src.encoder.gemma import GemmaStateEncoder

    model = _tiny_gpt2().to("cuda").half()
    enc = GemmaStateEncoder(
        feature_dim=16, max_length=None, device="cuda", dtype=torch.float16
    )
    enc._model, enc._tokenizer = model, _IdsTokenizer()

    out = enc.encode(["state"])

    assert tuple(out.shape) == (1, 16)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
