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
loop is unrolled by four, which turns the eight A loads for four consecutive K
steps into eight `float4` loads instead of thirty-two scalar loads: sixteen
shared loads per four K steps rather than forty. Only the FMA work is
irreducible, so trimming the load instructions is what raises the issue-bound
ceiling of the loop.

For `sm_80` and newer, aligned 16-byte `cp.async` instructions load the next
K-tile while the current tile is being consumed. A two-stage shared-memory
buffer prevents overwrite hazards. A is padded in shared memory and lane rows
are interleaved to avoid shared-memory bank conflicts. Results are written as
two `float4` vectors per owned row with streaming stores, which keeps C from
evicting the A/B strips out of L2.

The dispatch policy is:

| Shape condition | CTA tile | Threads | Path |
|---|---:|---:|---|
| `M >= 2N`, aligned | 128x64x8 | 128 | fast |
| `N >= 2M`, aligned | 64x128x8 | 128 | fast |
| balanced, aligned | 128x128x8 | 256 | fast |
| large, unaligned, `tiles(128) >= 24` | 128x128x8 | 256 | tail |
| large, unaligned, one side `>= 128` | 128x64x8 / 64x128x8 | 128 | tail |
| medium, unaligned | 64x64x8 | 64 | tail |
| too small to fill the device | 16x16x16 | 256 | edge |

"Aligned" for the fast path means complete CTA tiles and `K % 8 == 0`. The tail
kernels accept any positive M/N, any K and any alignment. They share the fast
kernel's inner loop: shared-memory tiles are zero filled outside the matrix, so
the K loop stays branch free and a K that is not a multiple of `BK` needs no
special case. Only the global-to-shared copy and the epilogue carry bounds
checks, and both run once per tile rather than once per FMA. This is what keeps
shapes such as `1000x1000x1000` off the scalar edge kernel.

The tail kernels also pick up shapes the fast predicates used to reject purely
because `K % 8 != 0`; for example `4096x4096x1001` now runs on a tiled kernel
instead of the 16x16 scalar path.

The edge kernel performs guarded scalar loads with one output element per
thread. It has a single serial FMA chain and no register reuse, so it only wins
when the tiled kernels would leave most of the device idle.

## Verifying the tiling without a GPU

`scripts/verify_tiling.py` mirrors the index arithmetic of both tiled kernels in
Python and checks four things that a GPU is not needed for:

1. every output element is written exactly once by exactly one thread;
2. every 16-byte shared and global access is 16-byte aligned;
3. the guarded shared-memory tile equals the zero-padded A/B block;
4. the 8x8 register accumulation reproduces `alpha * A @ B + beta * C`.

```bash
python3 scripts/verify_tiling.py
```

It cannot check `cp.async` or barrier placement; those still need a real GPU run.


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
