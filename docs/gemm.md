# CUDA FP32 GEMM: implementation and reproduction

## Contract

The public entry point is `fp32_gemm::launch` in `include/gemm.hpp`. Matrices are
dense and row-major:

```text
A: [M, K], B: [K, N], C: [M, N]
C = alpha * A * B + beta * C
```

The implementation intentionally uses FP32 CUDA Core FMA rather than WMMA/TF32.
The cuBLAS baseline is therefore configured with `CUBLAS_PEDANTIC_MATH` by
default.

## Kernel hierarchy

The fast path maps work at three levels:

```text
CTA tile (128x128, 128x64, or 64x128)
  -> warp tile (64x32)
    -> lane register tile (8x8)
```

Each lane holds 64 accumulators. For every K step it loads eight A values and
eight B values from shared memory, then performs an 8x8 outer product. The K
loop is unrolled.

For `sm_80` and newer, aligned 16-byte `cp.async` instructions load the next
K-tile while the current tile is being consumed. A two-stage shared-memory
buffer prevents overwrite hazards. A is padded in shared memory and lane rows
are interleaved to avoid shared-memory bank conflicts. Results are written as
two `float4` vectors per owned row.

The dispatch policy is:

| Shape condition | CTA tile | Threads | Path |
|---|---:|---:|---|
| `M >= 2N`, aligned | 128x64x8 | 128 | fast |
| `N >= 2M`, aligned | 64x128x8 | 128 | fast |
| balanced, aligned | 128x128x8 | 256 | fast |
| any other valid shape | 16x16x16 | 256 | edge |

The fast predicates require complete CTA tiles, `K % 8 == 0`, and vector-safe
row strides. The edge kernel performs guarded scalar loads and supports all
positive M/N and non-negative K without padding.

## Correctness and timing

`benchmarks/gemm_benchmark.cu` validates the nontrivial case
`alpha=1.25, beta=-0.5` against `cublasSgemm`. It then times both implementations
with CUDA Events on the same stream and reports
`2*M*N*K / elapsed_time` in GFLOP/s. The full suite contains exactly 38 unique
square, tall, wide, LLM-like, and odd edge shapes. Its final row reports the
arithmetic mean of per-shape cuBLAS percentages and their population standard
deviation in percentage points.

Useful commands:

```bash
make test CUDA_ARCH=sm_80
./build/gemm_benchmark --quick --iters 20
./scripts/profile_gemm.sh
./scripts/profile_gemm.sh --tf32
```

To inspect the asynchronous pipeline and memory behavior on a real GPU:

```bash
ncu --set full --kernel-name regex:gemm_fast_kernel \
    ./build/gemm_benchmark --quick --no-check --iters 5

nsys profile --stats=true -o results/gemm_timeline \
    ./build/gemm_benchmark --quick --no-check --iters 20
```

Watch `sm__pipe_fma_cycles_active`, shared-memory bank conflicts, achieved
occupancy, DRAM throughput, and eligible warps per cycle. Always rebuild with
the actual GPU architecture (for example `sm_86` for an RTX 3060).
