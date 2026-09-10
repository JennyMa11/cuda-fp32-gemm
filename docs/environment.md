# Environment Check

Checked on 2026-06-22.

## Available

- `nvcc`: CUDA compilation tools 12.8, V12.8.93.
- `g++-12`: available and used as the default CUDA host compiler.
- `g++`: system default is GCC 13.3.0, which CUDA 12.8 rejects by default.

## Missing or Blocked

- `cmake`: not installed.
- `nvidia-smi`: fails with GPU access blocked by the operating system.
- WSL GPU device: `/dev/dxg` is missing, so the Linux environment cannot access the Windows GPU.
- CUDA runtime execution: the sample detects unavailable GPU runtime and falls back to CPU validation.

## Result

Compilation works with:

```bash
make
```

`make run` now passes through CPU fallback in this restricted environment. Real GPU execution still needs a compatible NVIDIA driver and accessible WSL GPU runtime.
