#include "gemm.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace fp32_gemm {
namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ void copy_16b_async(float* dst, const float* src) {
#if __CUDA_ARCH__ >= 800
    const std::uint32_t smem = static_cast<std::uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(src));
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

// Each warp owns a 64x32 output tile. Its 32 lanes are arranged as 8x4;
// every lane accumulates an 8x8 register tile with outer products.
template <int BM, int BN, int BK, bool SIMPLE_EPILOGUE>
__global__ __launch_bounds__((BM / 64) * (BN / 32) * kWarpSize)
void gemm_fast_kernel(const float* __restrict__ a,
                      const float* __restrict__ b,
                      float* __restrict__ c,
                      int m,
                      int n,
                      int k,
                      float alpha,
                      float beta) {
    static_assert(BK == 8, "the vectorized pipeline expects BK=8");
    static_assert(BM % 64 == 0 && BN % 32 == 0, "invalid warp tiling");

    constexpr int kWarpsN = BN / 32;
    constexpr int kThreads = (BM / 64) * kWarpsN * kWarpSize;
    constexpr int kAPad = 4;

    __shared__ __align__(16) float smem_a[2][BM][BK + kAPad];
    __shared__ __align__(16) float smem_b[2][BK][BN];

    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;
    const int lane_m = lane / 4;
    const int lane_n = lane % 4;
    // Interleave the eight rows owned by a lane. Consecutive lane_m values then
    // hit different banks in smem_a (the four lane_n peers use broadcast).
    const int thread_row = warp_m * 64 + lane_m;
    const int thread_col = warp_n * 32 + lane_n * 8;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float accum[8][8] = {};

    auto load_tile = [&](int tile_k, int stage) {
        constexpr int kAFloat4 = BM * BK / 4;
        constexpr int kBFloat4 = BK * BN / 4;
        for (int vec = tid; vec < kAFloat4; vec += kThreads) {
            const int row = vec / (BK / 4);
            const int col = (vec % (BK / 4)) * 4;
            copy_16b_async(&smem_a[stage][row][col],
                           &a[(block_row + row) * k + tile_k * BK + col]);
        }
        for (int vec = tid; vec < kBFloat4; vec += kThreads) {
            const int row = vec / (BN / 4);
            const int col = (vec % (BN / 4)) * 4;
            copy_16b_async(&smem_b[stage][row][col],
                           &b[(tile_k * BK + row) * n + block_col + col]);
        }
    };

    const int tiles = k / BK;
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

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float ar[8];
            float br[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                ar[i] = smem_a[stage][thread_row + i * 8][kk];
            }
            *reinterpret_cast<float4*>(&br[0]) =
                *reinterpret_cast<const float4*>(&smem_b[stage][kk][thread_col]);
            *reinterpret_cast<float4*>(&br[4]) =
                *reinterpret_cast<const float4*>(&smem_b[stage][kk][thread_col + 4]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    accum[i][j] = fmaf(ar[i], br[j], accum[i][j]);
                }
            }
        }

        if (tile + 1 < tiles) {
            async_wait_all();
            __syncthreads();
        }
    }

    // Fast dispatch guarantees full tiles and 16-byte-aligned rows/columns.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float* out = &c[(block_row + thread_row + i * 8) * n + block_col + thread_col];
#pragma unroll
        for (int j = 0; j < 8; j += 4) {
            float4 value;
            if constexpr (SIMPLE_EPILOGUE) {
                value = make_float4(accum[i][j], accum[i][j + 1],
                                    accum[i][j + 2], accum[i][j + 3]);
            } else {
                const float4 old = *reinterpret_cast<const float4*>(out + j);
                value.x = alpha * accum[i][j] + beta * old.x;
                value.y = alpha * accum[i][j + 1] + beta * old.y;
                value.z = alpha * accum[i][j + 2] + beta * old.z;
                value.w = alpha * accum[i][j + 3] + beta * old.w;
            }
            *reinterpret_cast<float4*>(out + j) = value;
        }
    }
}

// A compact, bounds-checked path for arbitrary M/N/K. Keeping checks out of the
// fast kernel makes aligned production shapes measurably cheaper.
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

template <int BM, int BN>
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
    if (alpha == 1.0f && beta == 0.0f) {
        gemm_fast_kernel<BM, BN, 8, true><<<grid, block, 0, stream>>>(
            a, b, c, m, n, k, alpha, beta);
    } else {
        gemm_fast_kernel<BM, BN, 8, false><<<grid, block, 0, stream>>>(
            a, b, c, m, n, k, alpha, beta);
    }
    return {kind, block, grid};
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

    if (m >= 2 * n && m % 128 == 0 && n % 64 == 0 && k > 0 && k % 8 == 0) {
        return launch_fast<128, 64>(a, b, c, m, n, k, alpha, beta, stream,
                                    KernelKind::kFast128x64);
    }
    if (n >= 2 * m && m % 64 == 0 && n % 128 == 0 && k > 0 && k % 8 == 0) {
        return launch_fast<64, 128>(a, b, c, m, n, k, alpha, beta, stream,
                                    KernelKind::kFast64x128);
    }
    if (m % 128 == 0 && n % 128 == 0 && k > 0 && k % 8 == 0) {
        return launch_fast<128, 128>(a, b, c, m, n, k, alpha, beta, stream,
                                     KernelKind::kFast128x128);
    }

    const dim3 block(16, 16);
    const dim3 grid((n + 15) / 16, (m + 15) / 16);
    gemm_edge_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    return {KernelKind::kEdge, block, grid};
}

const char* kernel_name(KernelKind kind) {
    switch (kind) {
        case KernelKind::kFast128x128: return "fast_128x128x8";
        case KernelKind::kFast128x64: return "fast_128x64x8";
        case KernelKind::kFast64x128: return "fast_64x128x8";
        case KernelKind::kEdge: return "edge_16x16";
    }
    return "unknown";
}

}  // namespace fp32_gemm
