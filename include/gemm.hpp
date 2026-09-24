#pragma once

#include <cuda_runtime_api.h>

namespace fp32_gemm {

enum class KernelKind {
    kFast128x128,
    kFast128x64,
    kFast64x128,
    kTail128x128,
    kTail128x64,
    kTail64x128,
    kTail64x64,
    kEdge,
};

struct DispatchResult {
    KernelKind kind;
    dim3 block;
    dim3 grid;
};

// Row-major GEMM: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N].
// The matrices are dense (lda=K, ldb=N, ldc=N).
DispatchResult launch(const float* a,
                      const float* b,
                      float* c,
                      int m,
                      int n,
                      int k,
                      float alpha = 1.0f,
                      float beta = 0.0f,
                      cudaStream_t stream = nullptr);

const char* kernel_name(KernelKind kind);

}  // namespace fp32_gemm
