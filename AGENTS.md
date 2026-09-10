# Repository Guidelines

## Project Structure & Module Organization

This repository is for practicing CUDA syntax, kernels, and small operator implementations. Keep implementation code under `src/`; the initial example is `src/vector_add.cu`. Put correctness tests in `tests/`, performance experiments in `benchmarks/`, notes in `docs/`, and reusable helper scripts in `scripts/`. Build products belong in `build/` and should not be committed.

## Build, Test, and Development Commands

- `make` or `make vector_add` builds the sample CUDA program into `build/vector_add`.
- `make run` builds and runs the vector addition example. It uses the CUDA path when a GPU is available and CPU fallback when the runtime is unavailable.
- `make CUDA_ARCH=sm_86` overrides the default CUDA architecture when your GPU is not `sm_80`.
- `make HOST_CXX=g++-12` selects the CUDA host compiler; this is the default because CUDA 12.8 rejects GCC 13 without an override flag.
- `make env` prints CUDA, host compiler, and GPU driver information.
- `make clean` removes local build outputs.

`cmake` is not required for the current project skeleton. If a CMake build is added later, keep Makefile and README instructions in sync.

## Coding Style & Naming Conventions

Use C++17-compatible CUDA code. Prefer `.cu` for CUDA translation units and `.cuh` or `.hpp` for shared headers. Name kernels with a `_kernel` suffix, for example `vector_add_kernel`, and use descriptive operator names such as `reduce_sum` or `matmul_tiled`. Keep indentation consistent at 4 spaces, avoid unrelated refactors, and keep examples small enough to explain.

## Testing Guidelines

Add correctness tests under `tests/` and name files after the behavior being checked, such as `test_vector_add.cu`. For each new operator, include simple edge cases and at least one size large enough to exercise multiple blocks. Keep benchmarks separate from pass/fail tests and record GPU model, CUDA version, input shape, and timing method when reporting performance.

## Commit & Pull Request Guidelines

This repository has no established commit history yet. Use concise imperative commit subjects, such as `Add vector add example` or `Implement tiled matmul`. Pull requests should describe the operator or CUDA concept covered, list validation commands run, and mention GPU architecture assumptions or runtime limitations.

## Environment Notes

The current workspace has `nvcc` 12.8 and `g++-12` available. `cmake` is not installed, and `nvidia-smi` may fail if the execution environment blocks GPU access. Current examples should handle unavailable GPU runtime gracefully, but real CUDA performance work still requires successful GPU execution.

This WSL2 workspace can access the NVIDIA GPU, but CUDA commands must prioritize the WSL driver library; otherwise the runtime may report that no GPU is visible. Set the path before every GPU build, test, or benchmark session:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

This selects the WSL-provided `libcuda.so` instead of an incompatible Linux driver library. It does not bypass sandbox or permission controls.
