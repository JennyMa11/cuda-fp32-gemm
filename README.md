# CUDA Practice

This repository is a small workspace for practicing CUDA syntax, kernels, and operator implementations.

## Quick Start

```bash
make
make run
```

The default build compiles `src/vector_add.cu` with `nvcc`. `make run` uses the CUDA path when a GPU is available and falls back to CPU validation when the runtime is unavailable. If your GPU requires a different architecture, override it:

```bash
make CUDA_ARCH=sm_86
```

CUDA 12.8 rejects GCC 13 by default, so the Makefile uses `g++-12` as the host compiler. Override it when needed:

```bash
make HOST_CXX=/path/to/g++
```

Check the local toolchain and GPU visibility with:

```bash
make env
```

## Layout

- `src/` contains CUDA/C++ examples and operators.
- `tests/` is reserved for correctness tests.
- `benchmarks/` is reserved for performance experiments.
- `docs/` is reserved for notes and learning records.
- `scripts/` is reserved for helper scripts.

## FP32 GEMM project

The GEMM implementation computes dense row-major
`C = alpha * A * B + beta * C`. It contains three shape-aware fast kernels,
an arbitrary-size edge kernel, and a cuBLAS benchmark/validation harness.

```bash
# Ampere (change CUDA_ARCH for another GPU)
make gemm CUDA_ARCH=sm_86

# Four-shape correctness smoke test
make test CUDA_ARCH=sm_86

# Full 38-shape benchmark and CSV output
CUDA_ARCH=sm_86 make gemm
WARMUP=5 ITERS=20 ./scripts/profile_gemm.sh
```

The benchmark defaults to pedantic FP32 cuBLAS so the baseline does not silently
use TF32 Tensor Cores. Pass `--tf32` to the benchmark script to allow the default
cuBLAS math mode. Report measured results together with the GPU, clock/power
state, CUDA version, architecture flag, warmup count, and iteration count; the
ratio in a resume is not expected to transfer unchanged between GPUs.

Implementation details and profiling suggestions are in
[`docs/gemm.md`](docs/gemm.md).
