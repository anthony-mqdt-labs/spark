from __future__ import annotations

from spark.cli.download import _infer, _slug


def test_slug():
    assert _slug("mlx-community/Qwen2.5-0.5B-Instruct-4bit") == "qwen2.5-0.5b-instruct-4bit"


def test_infer_gguf():
    d = _infer("bartowski/Qwen2.5-7B-Instruct-GGUF")
    assert d["model_format"] == "gguf"
    assert d["params_billions"] == 7.0


def test_infer_mlx_quant_bit_style():
    d = _infer("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    assert d["model_format"] == "mlx"
    assert d["quant"] == "q4"
    assert d["params_billions"] == 0.5


def test_infer_vlm_detected():
    d = _infer("mlx-community/Qwen2-VL-7B-Instruct-4bit")
    assert d["model_format"] == "mlx-vlm"


def test_infer_unknown_format():
    d = _infer("someorg/mystery-model")
    assert d["model_format"] == "any"
