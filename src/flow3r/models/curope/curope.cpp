#include <torch/extension.h>
#include <cmath>

// Forward declaration of the CUDA launcher (rope_2d_cuda.cu)
void rope_2d_cuda(
    torch::Tensor tokens,
    const torch::Tensor positions,
    const float base,
    const float fwd);

// ---------------------------------------------------------------------------
// CPU reference implementation
//
// Token layout (Q = D/4):
//   axis 0 (Y): dims [0, Q)  -> u_Y,  dims [Q, 2Q) -> v_Y
//   axis 1 (X): dims [2Q,3Q) -> u_X,  dims [3Q,4Q) -> v_X
// ---------------------------------------------------------------------------

static void rope_2d_cpu(
    torch::Tensor tokens,
    const torch::Tensor positions,
    const float base,
    const float fwd)
{
    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int Q = tokens.size(3) / 4;   // quarter of D

    auto tok = tokens.accessor<float, 4>();
    auto pos = positions.accessor<int64_t, 3>();

    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            for (int axis = 0; axis < 2; ++axis) {
                const int p        = static_cast<int>(pos[b][n][axis]);
                const int u_offset = axis * 2 * Q;   // start of u slice for this axis

                for (int d = 0; d < Q; ++d) {
                    const float inv_freq = fwd * p / powf(base, d / float(Q));
                    const float c = cosf(inv_freq);
                    const float s = sinf(inv_freq);

                    for (int h = 0; h < H; ++h) {
                        const float u = tok[b][n][h][u_offset + d];
                        const float v = tok[b][n][h][u_offset + d + Q];

                        tok[b][n][h][u_offset + d]     = u * c - v * s;
                        tok[b][n][h][u_offset + d + Q] = v * c + u * s;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch: route to CUDA or CPU based on tensor device
// ---------------------------------------------------------------------------

void rope_2d(
    torch::Tensor tokens,           // (B, N, H, D),  D % 4 == 0
    const torch::Tensor positions,  // (B, N, 2),     int64
    const float base,
    const float fwd)
{
    // Shape / device checks
    TORCH_CHECK(tokens.dim()    == 4, "tokens must be 4-D (B,N,H,D)");
    TORCH_CHECK(positions.dim() == 3, "positions must be 3-D (B,N,2)");
    TORCH_CHECK(tokens.size(0) == positions.size(0),
        "batch size mismatch between tokens and positions");
    TORCH_CHECK(tokens.size(1) == positions.size(1),
        "sequence length mismatch between tokens and positions");
    TORCH_CHECK(positions.size(2) == 2,
        "positions.shape[2] must be 2 (Y and X axes)");
    TORCH_CHECK(tokens.size(3) % 4 == 0,
        "token dimension D must be divisible by 4");
    TORCH_CHECK(tokens.is_cuda() == positions.is_cuda(),
        "tokens and positions must be on the same device");

    if (tokens.is_cuda())
        rope_2d_cuda(tokens, positions, base, fwd);
    else
        rope_2d_cpu(tokens, positions, base, fwd);
}

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_2d", &rope_2d,
          "Apply 2D Rotary Position Embeddings in-place.\n\n"
          "Args:\n"
          "  tokens    (Tensor): float tensor of shape (B, N, H, D), D % 4 == 0\n"
          "  positions (Tensor): int64 tensor of shape (B, N, 2) — (y, x) coords\n"
          "  base      (float): frequency base (e.g. 10000)\n"
          "  fwd       (float): scale factor (1.0 for forward, -1.0 for inverse)");
}
