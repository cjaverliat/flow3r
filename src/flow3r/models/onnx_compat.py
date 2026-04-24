"""Dual-path toggle for ONNX export.

The default PyTorch inference path is unchanged. When code is executed inside
``onnx_export_mode()``, modules that use non-traceable constructs (e.g.
``nn.attention.sdpa_kernel`` context managers, Python-level caches keyed on
tensor shapes, data-dependent control flow over SymInts) take an equivalent
ONNX-friendly branch instead.
"""

from contextlib import contextmanager


_ONNX_EXPORT_MODE: bool = False


def is_onnx_export_mode() -> bool:
    return _ONNX_EXPORT_MODE


@contextmanager
def onnx_export_mode():
    global _ONNX_EXPORT_MODE
    previous = _ONNX_EXPORT_MODE
    _ONNX_EXPORT_MODE = True
    try:
        yield
    finally:
        _ONNX_EXPORT_MODE = previous
