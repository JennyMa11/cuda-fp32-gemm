#include "gemm.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace fp32_gemm {
namespace {

constexpr int kWarpSize = 32;
constexpr int kBK = 8;

// A is padded so the row stride stays a multiple of four floats (16 B, which the
// float4 loads and cp.async copies need) and so the eight rows one lane owns land
// on eight distinct banks.
constexpr int kAPad = 4;

// Below this many CTAs the largest guarded tile would leave the device idle, so
// the dispatcher steps down to a smaller tile or to the compact edge kernel.
constexpr int kMinTiles = 24;

__device__ __forceinline__ void copy_16b_async(float* dst, const float* src) {
#if __CUDA_ARCH__ >= 800
    const std::uint32_t smem = static_cast<std::uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(src));
#else
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
#endif
}

__device__ __forceinline__ void async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void async_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void zero_16b(float* dst) {
    *reinterpret_cast<float4*>(dst) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// Each lane owns an 8x8 register tile and is arranged as 8x4 inside its warp.
// Rows are interleaved (lane row r, then r + 8, ...) so that the eight lanes of
// a warp read eight consecutive smem rows and therefore eight distinct banks.
//
// The K loop is unrolled by four so the eight A values a lane needs for four
// consecutive k steps become two float4 loads instead of eight scalar loads:
// sixteen shared loads per four k steps rather than forty. That is what lifts
// the issue-bound ceiling of the loop, since only FMA work is irreducible.
template <int BM, int BN>
__device__ __forceinline__ void accumulate_tile(const float* __restrict__ smem_a,
                                                const float* __restrict__ smem_b,
                                                int thread_row,
                                                int thread_col,
                                                float (&accum)[8][8]) {
    static_assert(kBK % 4 == 0, "the vectorized K unroll needs BK to be a multiple of four");
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 4) {
        float ar[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 value = *reinterpret_cast<const float4*>(
                smem_a + (thread_row + i * 8) * (kBK + kAPad) + kk);
            ar[i][0] = value.x;
            ar[i][1] = value.y;
            ar[i][2] = value.z;
            ar[i][3] = value.w;
        }
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
            float br[8];
            *reinterpret_cast<float4*>(&br[0]) =
                *reinterpret_cast<const float4*>(smem_b + (kk + k2) * BN + thread_col);
            *reinterpret_cast<float4*>(&br[4]) =
                *reinterpret_cast<const float4*>(smem_b + (kk + k2) * BN + thread_col + 4);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    accum[i][j] = fmaf(ar[i][k2], br[j], accum[i][j]);
                }
            }
        }
    }
}

// Fast path. The dispatcher guarantees complete CTA tiles and vector-safe row
// strides, so this kernel contains no bounds check at all.
template <int BM, int BN, bool SIMPLE_EPILOGUE>
__global__ __launch_bounds__((BM / 64) * (BN / 32) * kWarpSize) void gemm_fast_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k,
    float alpha,
    float beta) {
    static_assert(BM % 64 == 0 && BN % 32 == 0, "invalid warp tiling");

    constexpr int kWarpsN = BN / 32;
    constexpr int kThreads = (BM / 64) * kWarpsN * kWarpSize;

    __shared__ __align__(16) float smem_a[2][BM][kBK + kAPad];
    __shared__ __align__(16) float smem_b[2][kBK][BN];

    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;
    const int lane_m = lane / 4;
    const int lane_n = lane % 4;
    const int thread_row = warp_m * 64 + lane_m;
    const int thread_col = warp_n * 32 + lane_n * 8;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float accum[8][8] = {};

    auto load_tile = [&](int tile_k, int stage) {
        constexpr int kAFloat4 = BM * kBK / 4;
        constexpr int kBFloat4 = kBK * BN / 4;
        for (int vec = tid; vec < kAFloat4; vec += kThreads) {
            const int row = vec / (kBK / 4);
            const int col = (vec % (kBK / 4)) * 4;
            copy_16b_async(&smem_a[stage][row][col],
                           &a[(block_row + row) * k + tile_k * kBK + col]);
        }
        for (int vec = tid; vec < kBFloat4; vec += kThreads) {
            const int row = vec / (BN / 4);
            const int col = (vec % (BN / 4)) * 4;
            copy_16b_async(&smem_b[stage][row][col],
                           &b[(tile_k * kBK + row) * n + block_col + col]);
        }
    };

    const int tiles = k / kBK;
    load_tile(0, 0);
    async_commit();
    async_wait_all();
    __syncthreads();

    for (int tile = 0; tile < tiles; ++tile) {
        const int stage = tile & 1;
        if (tile + 1 < tiles) {
            load_tile(tile + 1, stage ^ 1);
            async_commit();
        }

        accumulate_tile<BM, BN>(&smem_a[stage][0][0], &smem_b[stage][0][0], thread_row,
                                thread_col, accum);

        if (tile + 1 < tiles) {
            async_wait_all();
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float* out = &c[(block_row + thread_row + i * 8) * n + block_col + thread_col];
#pragma unroll
        for (int j = 0; j < 8; j += 4) {
            float4 value;
            if constexpr (SIMPLE_EPILOGUE) {
                value = make_float4(accum[i][j], accum[i][j + 1], accum[i][j + 2],
                                    accum[i][j + 3]);
            } else {
                const float4 old = *reinterpret_cast<const float4*>(out + j);
                value.x = alpha * accum[i][j] + beta * old.x;
                value.y = alpha * accum[i][j + 1] + beta * old.y;
                value.z = alpha * accum[i][j + 2] + beta * old.z;
                value.w = alpha * accum[i][j + 3] + beta * old.w;
            }
            // Streaming stores keep C from evicting the A/B strips out of L2.
            __stcs(reinterpret_cast<float4*>(out + j), value);
        }
    }
}

// Guarded path for every shape the fast predicates reject: arbitrary M/N, any K,
// any alignment. Shared-memory tiles are zero filled outside the matrix, so the
// K loop keeps the same branch-free body as the fast kernel and a K that is not
// a multiple of BK needs no special case -- the zero padding contributes zero to
// the accumulation. Only the global -> shared copy and the epilogue are guarded,
// and both run once per tile rather than once per FMA.
//
// VEC is true only when K and N are multiples of four, which is what the 16-byte
// global accesses and the vector epilogue need; otherwise the scalar path is used.
template <int BM, int BN, bool SIMPLE_EPILOGUE, bool VEC>
__global__ __launch_bounds__((BM / 64) * (BN / 32) * kWarpSize) void gemm_tail_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k,
    float alpha,
    float beta) {
    static_assert(BM % 64 == 0 && BN % 32 == 0, "invalid warp tiling");

    constexpr int kWarpsN = BN / 32;
    constexpr int kThreads = (BM / 64) * kWarpsN * kWarpSize;

    __shared__ __align__(16) float smem_a[2][BM][kBK + kAPad];
    __shared__ __align__(16) float smem_b[2][kBK][BN];

    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;
    const int lane_m = lane / 4;
    const int lane_n = lane % 4;
    const int thread_row = warp_m * 64 + lane_m;
    const int thread_col = warp_n * 32 + lane_n * 8;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float accum[8][8] = {};

    auto load_tile = [&](int tile_k, int stage) {
        if constexpr (VEC) {
            constexpr int kAFloat4 = BM * kBK / 4;
            for (int vec = tid; vec < kAFloat4; vec += kThreads) {
                const int row = vec / (kBK / 4);
                const int col = (vec % (kBK / 4)) * 4;
                float* dst = &smem_a[stage][row][col];
                const int grow = block_row + row;
                const int gcol = tile_k * kBK + col;
                if (grow < m && gcol + 4 <= k) {
                    copy_16b_async(dst, &a[grow * k + gcol]);
                } else {
                    zero_16b(dst);
                }
            }
            constexpr int kBFloat4 = kBK * BN / 4;
            for (int vec = tid; vec < kBFloat4; vec += kThreads) {
                const int row = vec / (BN / 4);
                const int col = (vec % (BN / 4)) * 4;
                float* dst = &smem_b[stage][row][col];
                const int grow = tile_k * kBK + row;
                const int gcol = block_col + col;
                if (grow < k && gcol + 4 <= n) {
                    copy_16b_async(dst, &b[grow * n + gcol]);
                } else {
                    zero_16b(dst);
                }
            }
        } else {
            for (int idx = tid; idx < BM * kBK; idx += kThreads) {
                const int row = idx / kBK;
                const int col = idx % kBK;
                const int grow = block_row + row;
                const int gcol = tile_k * kBK + col;
                smem_a[stage][row][col] = (grow < m && gcol < k) ? a[grow * k + gcol] : 0.0f;
            }
            for (int idx = tid; idx < kBK * BN; idx += kThreads) {
                const int row = idx / BN;
                const int col = idx % BN;
                const int grow = tile_k * kBK + row;
                const int gcol = block_col + col;
                smem_b[stage][row][col] = (grow < k && gcol < n) ? b[grow * n + gcol] : 0.0f;
            }
        }
    };

    const int tiles = (k + kBK - 1) / kBK;
    load_tile(0, 0);
    async_commit();
    async_wait_all();
    __syncthreads();

    for (int tile = 0; tile < tiles; ++tile) {
        const int stage = tile & 1;
        if (tile + 1 < tiles) {
            load_tile(tile + 1, stage ^ 1);
            async_commit();
        }

        accumulate_tile<BM, BN>(&smem_a[stage][0][0], &smem_b[stage][0][0], thread_row,
                                thread_col, accum);

        if (tile + 1 < tiles) {
            async_wait_all();
            __syncthreads();
        }
    }

    const int rows_left = m - block_row;
    const int cols_left = n - block_col;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int local_row = thread_row + i * 8;
        if (local_row >= rows_left) {
            break;
        }
        float* out =
            &c[static_cast<std::size_t>(block_row + local_row) * n + block_col + thread_col];
        if constexpr (VEC) {
            if (cols_left - thread_col >= 8) {
#pragma unroll
                for (int j = 0; j < 8; j += 4) {
                    float4 value;
                    if constexpr (SIMPLE_EPILOGUE) {
                        value = make_float4(accum[i][j], accum[i][j + 1], accum[i][j + 2],
                                            accum[i][j + 3]);
                    } else {
                        const float4 old = *reinterpret_cast<const float4*>(out + j);
                        value.x = alpha * accum[i][j] + beta * old.x;
                        value.y = alpha * accum[i][j + 1] + beta * old.y;
                        value.z = alpha * accum[i][j + 2] + beta * old.z;
                        value.w = alpha * accum[i][j + 3] + beta * old.w;
                    }
                    __stcs(reinterpret_cast<float4*>(out + j), value);
                }
                continue;
            }
        }
        const int limit = (cols_left - thread_col < 8) ? cols_left - thread_col : 8;
        for (int j = 0; j < limit; ++j) {
            out[j] = alpha * accum[i][j] + beta * out[j];
        }
    }
}

// Compact fallback for shapes too small to tile: one output element per thread,
// no register blocking. It only wins when there is not enough work to fill the
// device with the tiled kernels, so it stays deliberately simple.
__global__ void gemm_edge_kernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c,
                                 int m,
                                 int n,
                                 int k,
                                 float alpha,
                                 float beta) {
    constexpr int TILE = 16;
    __shared__ float as[TILE][TILE + 1];
    __shared__ float bs[TILE][TILE + 1];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int base = 0; base < k; base += TILE) {
        const int ak = base + threadIdx.x;
        const int bk = base + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = (row < m && ak < k) ? a[row * k + ak] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = (bk < k && col < n) ? b[bk * n + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            sum = fmaf(as[threadIdx.y][kk], bs[kk][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = alpha * sum + beta * c[row * n + col];
    }
}

template <int BM, int BN, bool SIMPLE_EPILOGUE>
DispatchResult launch_fast(const float* a,
                           const float* b,
                           float* c,
                           int m,
                           int n,
                           int k,
                           float alpha,
                           float beta,
                           cudaStream_t stream,
                           KernelKind kind) {
    constexpr int threads = (BM / 64) * (BN / 32) * kWarpSize;
    const dim3 block(threads);
    const dim3 grid(n / BN, m / BM);
    gemm_fast_kernel<BM, BN, SIMPLE_EPILOGUE><<<grid, block, 0, stream>>>(a, b, c, m, n, k,
                                                                         alpha, beta);
    return {kind, block, grid};
}

template <int BM, int BN, bool VEC>
DispatchResult launch_tail_tile(const float* a,
                                const float* b,
                                float* c,
                                int m,
                                int n,
                                int k,
                                float alpha,
                                float beta,
                                cudaStream_t stream,
                                KernelKind kind) {
    constexpr int threads = (BM / 64) * (BN / 32) * kWarpSize;
    const dim3 block(threads);
    const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    if (alpha == 1.0f && beta == 0.0f) {
        gemm_tail_kernel<BM, BN, true, VEC><<<grid, block, 0, stream>>>(a, b, c, m, n, k,
                                                                       alpha, beta);
    } else {
        gemm_tail_kernel<BM, BN, false, VEC><<<grid, block, 0, stream>>>(a, b, c, m, n, k,
                                                                        alpha, beta);
    }
    return {kind, block, grid};
}

DispatchResult launch_edge(const float* a,
                           const float* b,
                           float* c,
                           int m,
                           int n,
                           int k,
                           float alpha,
                           float beta,
                           cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid((n + 15) / 16, (m + 15) / 16);
    gemm_edge_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    return {KernelKind::kEdge, block, grid};
}

template <bool VEC>
DispatchResult launch_tail_auto(const float* a,
                                const float* b,
                                float* c,
                                int m,
                                int n,
                                int k,
                                float alpha,
                                float beta,
                                cudaStream_t stream) {
    const long long tiles_128 =
        static_cast<long long>((m + 127) / 128) * static_cast<long long>((n + 127) / 128);
    if (m >= 128 && n >= 128 && tiles_128 >= kMinTiles) {
        return launch_tail_tile<128, 128, VEC>(a, b, c, m, n, k, alpha, beta, stream,
                                               KernelKind::kTail128x128);
    }
    if (m >= 64 && n >= 64) {
        const long long tiles_64 =
            static_cast<long long>((m + 63) / 64) * static_cast<long long>((n + 63) / 64);
        if (tiles_64 >= kMinTiles / 2) {
            if (m >= 128) {
                return launch_tail_tile<128, 64, VEC>(a, b, c, m, n, k, alpha, beta, stream,
                                                      KernelKind::kTail128x64);
            }
            if (n >= 128) {
                return launch_tail_tile<64, 128, VEC>(a, b, c, m, n, k, alpha, beta, stream,
                                                      KernelKind::kTail64x128);
            }
            return launch_tail_tile<64, 64, VEC>(a, b, c, m, n, k, alpha, beta, stream,
                                                 KernelKind::kTail64x64);
        }
    }
    return launch_edge(a, b, c, m, n, k, alpha, beta, stream);
}

}  // namespace

DispatchResult launch(const float* a,
                      const float* b,
                      float* c,
                      int m,
                      int n,
                      int k,
                      float alpha,
                      float beta,
                      cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k < 0 || a == nullptr || b == nullptr || c == nullptr) {
        return {KernelKind::kEdge, dim3(0), dim3(0)};
    }

    // The fast kernels need complete CTA tiles and a K that is a multiple of BK.
    if (k > 0 && k % kBK == 0) {
        if (m >= 2 * n && m % 128 == 0 && n % 64 == 0) {
            return launch_fast<128, 64>(a, b, c, m, n, k, alpha, beta, stream,
                                        KernelKind::kFast128x64);
        }
        if (n >= 2 * m && m % 64 == 0 && n % 128 == 0) {
            return launch_fast<64, 128>(a, b, c, m, n, k, alpha, beta, stream,
                                        KernelKind::kFast64x128);
        }
        if (m % 128 == 0 && n % 128 == 0) {
            return launch_fast<128, 128>(a, b, c, m, n, k, alpha, beta, stream,
                                         KernelKind::kFast128x128);
        }
    }

    // VEC needs 16-byte aligned rows on both operands: A rows advance by K and B
    // rows advance by N.
    if (k % 4 == 0 && n % 4 == 0) {
        return launch_tail_auto<true>(a, b, c, m, n, k, alpha, beta, stream);
    }
    return launch_tail_auto<false>(a, b, c, m, n, k, alpha, beta, stream);
}

const char* kernel_name(KernelKind kind) {
    switch (kind) {
        case KernelKind::kFast128x128: return "fast_128x128x8";
        case KernelKind::kFast128x64: return "fast_128x64x8";
        case KernelKind::kFast64x128: return "fast_64x128x8";
        case KernelKind::kTail128x128: return "tail_128x128x8";
        case KernelKind::kTail128x64: return "tail_128x64x8";
        case KernelKind::kTail64x128: return "tail_64x128x8";
        case KernelKind::kTail64x64: return "tail_64x64x8";
        case KernelKind::kEdge: return "edge_16x16";
    }
    return "unknown";
}

}  // namespace fp32_gemm
