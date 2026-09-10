NVCC ?= nvcc
HOST_CXX ?= g++-12
CUDA_ARCH ?= sm_80
BUILD_DIR ?= build
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo -arch=$(CUDA_ARCH) -Xcompiler=-Wall

.PHONY: all vector_add gemm run run-gemm test clean env FORCE

all: vector_add gemm

vector_add: $(BUILD_DIR)/vector_add

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

FORCE:

# Rebuild when CUDA_ARCH changes even though the output filename stays stable.
$(BUILD_DIR)/.cuda_arch: FORCE | $(BUILD_DIR)
	@test -f $@ && test "$$(cat $@)" = "$(CUDA_ARCH)" || printf '%s\n' "$(CUDA_ARCH)" > $@

$(BUILD_DIR)/vector_add: src/vector_add.cu $(BUILD_DIR)/.cuda_arch | $(BUILD_DIR)
	$(NVCC) -ccbin $(HOST_CXX) $(NVCCFLAGS) $< -o $@

gemm: $(BUILD_DIR)/gemm_benchmark

$(BUILD_DIR)/gemm_benchmark: src/gemm.cu benchmarks/gemm_benchmark.cu include/gemm.hpp $(BUILD_DIR)/.cuda_arch | $(BUILD_DIR)
	$(NVCC) -ccbin $(HOST_CXX) $(NVCCFLAGS) -Iinclude src/gemm.cu benchmarks/gemm_benchmark.cu -lcublas -o $@

run: $(BUILD_DIR)/vector_add
	./$(BUILD_DIR)/vector_add

run-gemm: $(BUILD_DIR)/gemm_benchmark
	./$(BUILD_DIR)/gemm_benchmark

test: $(BUILD_DIR)/gemm_benchmark
	./$(BUILD_DIR)/gemm_benchmark --quick --warmup 1 --iters 3

env:
	$(NVCC) --version
	$(HOST_CXX) --version
	@command -v cmake >/dev/null 2>&1 && cmake --version || echo "cmake: not found"
	@nvidia-smi || true

clean:
	rm -rf $(BUILD_DIR)
