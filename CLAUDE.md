# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A CUDA practice workspace for learning syntax, kernels, and small operator implementations. Builds with `nvcc` and a plain Makefile (no cmake).

## Build & Run

```bash
make                    # compile src/vector_add.cu → build/vector_add
make run                # build and execute (GPU path or CPU fallback)
make CUDA_ARCH=sm_86   # override GPU architecture (default: sm_80)
make HOST_CXX=g++-12   # override host compiler (default: g++-12, required because CUDA 12.8 rejects GCC 13)
make env                # print nvcc, host compiler, cmake, and nvidia-smi versions
make clean              # remove build/
```

When adding a new `.cu` file, add a corresponding Makefile target following the `vector_add` pattern.

## Project Layout

- `src/` — CUDA/C++ source files (`.cu` for translation units, `.cuh`/`.hpp` for headers)
- `tests/` — correctness tests, named `test_<behavior>.cu`
- `benchmarks/` — performance experiments (separate from pass/fail tests)
- `docs/` — notes and learning records
- `scripts/` — helper scripts
- `build/` — build output (gitignored)

## Coding Conventions

- C++17-compatible CUDA code
- Kernel names use `_kernel` suffix (e.g., `vector_add_kernel`)
- Operator names are descriptive (e.g., `reduce_sum`, `matmul_tiled`)
- 4-space indentation
- Each example should be small enough to explain
- Use the `CUDA_CHECK` macro pattern from `src/vector_add.cu` for error handling

## Testing

- Tests go in `tests/` as `test_<name>.cu`
- Cover edge cases and at least one size large enough to exercise multiple blocks
- When reporting benchmarks, record GPU model, CUDA version, input shape, and timing method

## Environment Notes

- `nvcc` 12.8 with `g++-12` as host compiler
- `cmake` is not installed
- GPU access may be blocked (WSL environment); examples should handle unavailable GPU runtime gracefully with CPU fallback
