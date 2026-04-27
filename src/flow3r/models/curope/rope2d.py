import torch

def get_cos_sin(base, D, seq_len, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float().to(device) / D))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()

class RoPE2D(torch.nn.Module):

    def __init__(self, freq=100.0, F0=1.0, max_pos=512):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.max_pos = max_pos

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2

        if torch.jit.is_tracing() or torch.compiler.is_compiling():
            max_pos = torch.tensor(self.max_pos, dtype=torch.int64, device=tokens.device)
        else:
            max_pos = positions.max().long()

        cos, sin = get_cos_sin(self.base, D, max_pos + 1, tokens.device, tokens.dtype)

        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens