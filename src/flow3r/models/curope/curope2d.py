import torch

# Try installed wheel first, fall back to in-place build artefact.
try:
    import curope as _kernels          # pip / python setup.py install
except ModuleNotFoundError:
    from . import curope as _kernels   # python setup.py build_ext --inplace


class _RoPE2DFunction(torch.autograd.Function):
    """In-place 2-D RoPE as a differentiable torch.autograd.Function.

    The kernel mutates *tokens* directly (in-place).  The backward pass
    inverts the rotation by negating the frequency scale F0, which is
    mathematically equivalent to applying the transpose of the rotation
    matrix (R^T = R^{-1} for orthogonal matrices).
    """

    @staticmethod
    def forward(ctx, tokens, positions, base: float, F0: float):
        # Save only what the backward pass needs.
        ctx.save_for_backward(positions)
        ctx.base = base
        ctx.F0   = F0

        _kernels.rope_2d(tokens, positions, base, F0)
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_output):
        (positions,) = ctx.saved_tensors

        # Invert the rotation: negate the frequency scale.
        _kernels.rope_2d(grad_output, positions, ctx.base, -ctx.F0)
        ctx.mark_dirty(grad_output)

        # Gradients for: tokens, positions, base, F0
        return grad_output, None, None, None


class RoPE2D(torch.nn.Module):
    """Apply 2-D Rotary Position Embeddings to a token tensor.

    Args:
        freq (float): Frequency base used to build the inverse-frequency
                      schedule (analogous to ``base`` in the 1-D formulation).
        F0   (float): Forward scale factor.  Pass ``-1`` to invert.

    Forward signature::

        tokens    (B, N, H, D)  – float, modified **in-place**
        positions (B, N, 2)     – int64, (y, x) grid coordinates

    Returns the (modified) ``tokens`` tensor.
    """

    def __init__(self, freq: float = 100.0, F0: float = 1.0) -> None:
        super().__init__()
        self.base = freq
        self.F0   = F0

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # The kernel expects (B, N, H, D); tokens arrive as (B, H, N, D),
        # so we transpose the sequence and head axes before passing in.
        _RoPE2DFunction.apply(tokens.transpose(1, 2), positions, self.base, self.F0)
        return tokens