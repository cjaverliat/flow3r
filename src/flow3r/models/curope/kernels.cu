#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CHECK_CONTIGUOUS_CUDA(t) \
    TORCH_CHECK((t).is_cuda(),       #t " must be a CUDA tensor");   \
    TORCH_CHECK((t).is_contiguous(), #t " must be contiguous");

static void check_last_cuda_error() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// Kernel
//
// Token layout (D dims, Q = D/4):
//   [ u_Y | v_Y | u_X | v_X ]
//     0..Q  Q..2Q 2Q..3Q 3Q..4Q
//
// Rotation for axis a in {Y=0, X=1}, dimension d in [0, Q):
//   freq  = position[b,n,a] * fwd / base^(d / Q)
//   u'    = u*cos(freq) - v*sin(freq)
//   v'    = v*cos(freq) + u*sin(freq)
//
// Grid : (B*N) blocks
// Block: D threads  (one thread per token dimension)
// Shared: D floats for one head's data  +  Q floats for cached inv-freqs
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void rope_2d_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> tokens,
    const int64_t* __restrict__ positions,
    const float base,
    const float fwd)
{
    const int N = tokens.size(1);
    const int H = tokens.size(2);
    const int D = tokens.size(3);
    const int Q = D / 4;   // quarter size

    const int b = blockIdx.x / N;
    const int n = blockIdx.x % N;

    // Shared layout: [D floats: head scratch] [Q floats: inv_freq cache]
    extern __shared__ float smem[];
    float* head_buf  = smem;
    float* inv_freqs = smem + D;

    // Precompute inverse frequencies (only Q threads participate)
    if (threadIdx.x < Q)
        inv_freqs[threadIdx.x] = fwd / powf(base, threadIdx.x / float(Q));
    __syncthreads();

    // Which axis (Y=0 / X=1) and which dimension within that axis quarter
    const int axis = (threadIdx.x < D / 2) ? 0 : 1;
    const int d    = threadIdx.x % Q;   // index within quarter [0, Q)

    // Position for this axis
    const int pos = static_cast<int>(positions[(b * N + n) * 2 + axis]);

    // Rotation angle
    const float angle = pos * inv_freqs[d];
    const float c = cosf(angle);
    const float s = sinf(angle);

    // Index of the u component in the token for this (axis, d)
    const int u_idx = axis * (D / 2) + d;       // u
    const int v_idx = axis * (D / 2) + d + Q;   // v

    // Apply rotation for every head
    for (int h = 0; h < H; ++h) {
        // Load entire head slice into shared memory
        head_buf[threadIdx.x] = static_cast<float>(tokens[b][n][h][threadIdx.x]);
        __syncthreads();

        const float u = head_buf[u_idx];
        const float v = head_buf[v_idx];

        if ((threadIdx.x % (D / 2)) < Q)
            // u slot
            tokens[b][n][h][threadIdx.x] = static_cast<scalar_t>(u * c - v * s);
        else
            // v slot
            tokens[b][n][h][threadIdx.x] = static_cast<scalar_t>(v * c + u * s);

        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

void rope_2d_cuda(
    torch::Tensor tokens,
    const torch::Tensor positions,
    const float base,
    const float fwd)
{
    CHECK_CONTIGUOUS_CUDA(tokens);
    CHECK_CONTIGUOUS_CUDA(positions);

    const int B = tokens.size(0);
    const int N = tokens.size(1);
    const int D = tokens.size(3);

    TORCH_CHECK(D % 4 == 0,
        "token dimension D must be divisible by 4, got D=", D);
    TORCH_CHECK(positions.size(0) == B && positions.size(1) == N && positions.size(2) == 2,
        "positions must have shape (B, N, 2)");

    const int blocks        = B * N;
    const int threads       = D;
    const size_t shared_mem = sizeof(float) * (D + D / 4);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, tokens.scalar_type(), "rope_2d_cuda",
        ([&] {
            rope_2d_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
                tokens.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                positions.data_ptr<int64_t>(),
                base, fwd);
        }));

    check_last_cuda_error();
}
