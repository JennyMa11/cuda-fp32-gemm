#include "gemm.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        const cudaError_t status_ = (expr);                                     \
        if (status_ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(status_)          \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

#define CUBLAS_CHECK(expr)                                                      \
    do {                                                                        \
        const cublasStatus_t status_ = (expr);                                  \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS error " << static_cast<int>(status_)           \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

namespace {

struct Shape {
    int m;
    int n;
    int k;
};

struct Options {
    int warmup = 3;
    int iterations = 10;
    bool check = true;
    bool quick = false;
    bool tf32 = false;
    std::string csv;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    std::size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

__global__ void initialize_kernel(float* data, std::size_t count, int seed) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const unsigned int x = static_cast<unsigned int>(i) * 1664525u +
                               static_cast<unsigned int>(seed) * 1013904223u;
        data[i] = (static_cast<int>((x >> 8) & 0xffffu) - 32768) / 65536.0f;
    }
}

void initialize(float* data, std::size_t count, int seed, cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    initialize_kernel<<<blocks, threads, 0, stream>>>(data, count, seed);
    CUDA_CHECK(cudaGetLastError());
}

void cublas_row_major_gemm(cublasHandle_t handle,
                           const float* a,
                           const float* b,
                           float* c,
                           const Shape& s,
                           float alpha,
                           float beta) {
    // cuBLAS is column-major. Swapping A/B computes C^T = B^T A^T while all
    // allocations remain row-major and no explicit transpose is needed.
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             s.n,
                             s.m,
                             s.k,
                             &alpha,
                             b,
                             s.n,
                             a,
                             s.k,
                             &beta,
                             c,
                             s.n));
}

template <typename Launch>
float measure_ms(Launch launch, int warmup, int iterations, cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin, stream));
    for (int i = 0; i < iterations; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return elapsed / iterations;
}

bool validate(const float* actual,
              const float* expected,
              std::size_t count,
              float* max_abs_out,
              float* max_rel_out) {
    std::vector<float> h_actual(count);
    std::vector<float> h_expected(count);
    CUDA_CHECK(cudaMemcpy(h_actual.data(), actual, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_expected.data(), expected, count * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float max_ref = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        const float abs_error = std::abs(h_actual[i] - h_expected[i]);
        const float rel_error = abs_error / std::max(std::abs(h_expected[i]), 1.0e-3f);
        max_abs = std::max(max_abs, abs_error);
        max_rel = std::max(max_rel, rel_error);
        max_ref = std::max(max_ref, std::abs(h_expected[i]));
    }
    *max_abs_out = max_abs;
    *max_rel_out = max_rel;
    return max_abs <= 2.0e-4f * std::max(1.0f, max_ref);
}

std::vector<Shape> benchmark_shapes() {
    // 10 square + 23 aligned rectangular + 5 edge cases = 38 shapes.
    return {
        {128, 128, 128},       {256, 256, 256},       {384, 384, 384},
        {512, 512, 512},       {768, 768, 768},       {1024, 1024, 1024},
        {1536, 1536, 1536},    {2048, 2048, 2048},    {3072, 3072, 3072},
        {4096, 4096, 4096},
        {128, 4096, 4096},     {256, 4096, 4096},     {512, 4096, 4096},
        {1024, 4096, 4096},    {4096, 128, 4096},     {4096, 256, 4096},
        {4096, 512, 4096},     {4096, 1024, 4096},    {256, 11008, 4096},
        {512, 11008, 4096},    {1024, 11008, 4096},   {2048, 11008, 4096},
        {4096, 11008, 4096},   {256, 4096, 11008},    {512, 4096, 11008},
        {1024, 4096, 11008},   {2048, 4096, 11008},   {4096, 4096, 11008},
        {1024, 3072, 4096},    {2048, 3072, 4096},    {3072, 1024, 4096},
        {3072, 2048, 4096},    {4096, 3072, 11008},
        {127, 255, 63},        {511, 769, 257},        {1000, 1000, 1000},
        {1537, 1025, 769},     {33, 65, 17},
    };
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) {
            options.warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            options.iterations = std::stoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            options.csv = argv[++i];
        } else if (arg == "--no-check") {
            options.check = false;
        } else if (arg == "--quick") {
            options.quick = true;
        } else if (arg == "--tf32") {
            options.tf32 = true;
        } else if (arg == "--help") {
            std::cout << "Usage: gemm_benchmark [--quick] [--warmup N] [--iters N] "
                         "[--no-check] [--tf32] [--csv FILE]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
    if (options.warmup < 0 || options.iterations <= 0) {
        std::cerr << "warmup must be non-negative and iterations must be positive\n";
        std::exit(EXIT_FAILURE);
    }
    return options;
}

double gflops(const Shape& s, float milliseconds) {
    return 2.0 * static_cast<double>(s.m) * s.n * s.k / (milliseconds * 1.0e6);
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    int device_count = 0;
    const cudaError_t probe = cudaGetDeviceCount(&device_count);
    if (probe != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no CUDA device is visible";
        if (probe != cudaSuccess) {
            std::cout << " (" << cudaGetErrorString(probe) << ')';
        }
        std::cout << '\n';
        return 0;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, options.tf32 ? CUBLAS_DEFAULT_MATH
                                                        : CUBLAS_PEDANTIC_MATH));

    std::vector<Shape> shapes = benchmark_shapes();
    if (options.quick) {
        shapes = {{256, 256, 256}, {512, 1024, 768}, {1000, 1000, 1000}, {127, 255, 63}};
    }

    std::ofstream csv;
    if (!options.csv.empty()) {
        csv.open(options.csv);
        if (!csv) {
            std::cerr << "Cannot open CSV output: " << options.csv << '\n';
            return EXIT_FAILURE;
        }
        csv << "m,n,k,kernel,custom_ms,custom_gflops,cublas_ms,cublas_gflops,ratio_pct,valid,max_abs,max_rel\n";
    }

    std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n"
              << "cuBLAS math: " << (options.tf32 ? "default/TF32 allowed" : "pedantic FP32")
              << ", shapes: " << shapes.size() << '\n';
    std::cout << std::left << std::setw(18) << "M x N x K" << std::setw(20) << "kernel"
              << std::right << std::setw(11) << "ours ms" << std::setw(13) << "ours GF/s"
              << std::setw(12) << "cuBLAS ms" << std::setw(14) << "cuBLAS GF/s"
              << std::setw(10) << "ratio" << std::setw(9) << "check" << '\n';

    std::vector<double> ratios;
    bool all_valid = true;
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        const Shape s = shapes[index];
        const std::size_t a_count = static_cast<std::size_t>(s.m) * s.k;
        const std::size_t b_count = static_cast<std::size_t>(s.k) * s.n;
        const std::size_t c_count = static_cast<std::size_t>(s.m) * s.n;
        DeviceBuffer<float> a(a_count);
        DeviceBuffer<float> b(b_count);
        DeviceBuffer<float> c(c_count);
        DeviceBuffer<float> reference(c_count);
        initialize(a.get(), a_count, 11 + static_cast<int>(index), stream);
        initialize(b.get(), b_count, 29 + static_cast<int>(index), stream);
        initialize(c.get(), c_count, 47 + static_cast<int>(index), stream);
        CUDA_CHECK(cudaMemcpyAsync(reference.get(), c.get(), c_count * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));

        const float check_alpha = 1.25f;
        const float check_beta = -0.5f;
        const auto dispatch = fp32_gemm::launch(a.get(), b.get(), c.get(), s.m, s.n, s.k,
                                                 check_alpha, check_beta, stream);
        CUDA_CHECK(cudaGetLastError());
        cublas_row_major_gemm(handle, a.get(), b.get(), reference.get(), s,
                              check_alpha, check_beta);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float max_abs = 0.0f;
        float max_rel = 0.0f;
        const bool valid = !options.check || validate(c.get(), reference.get(), c_count,
                                                       &max_abs, &max_rel);
        all_valid = all_valid && valid;

        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        const float custom_ms = measure_ms(
            [&] { fp32_gemm::launch(a.get(), b.get(), c.get(), s.m, s.n, s.k,
                                     alpha, beta, stream); },
            options.warmup, options.iterations, stream);
        const float cublas_ms = measure_ms(
            [&] { cublas_row_major_gemm(handle, a.get(), b.get(), reference.get(), s,
                                         alpha, beta); },
            options.warmup, options.iterations, stream);
        const double custom_gflops = gflops(s, custom_ms);
        const double cublas_gflops = gflops(s, cublas_ms);
        const double ratio = 100.0 * custom_gflops / cublas_gflops;
        ratios.push_back(ratio);

        const std::string shape = std::to_string(s.m) + "x" + std::to_string(s.n) + "x" +
                                  std::to_string(s.k);
        std::cout << std::left << std::setw(18) << shape
                  << std::setw(20) << fp32_gemm::kernel_name(dispatch.kind) << std::right
                  << std::fixed << std::setprecision(3) << std::setw(11) << custom_ms
                  << std::setprecision(1) << std::setw(13) << custom_gflops
                  << std::setprecision(3) << std::setw(12) << cublas_ms
                  << std::setprecision(1) << std::setw(14) << cublas_gflops
                  << std::setprecision(1) << std::setw(9) << ratio << '%'
                  << std::setw(9) << (valid ? "PASS" : "FAIL") << '\n';

        if (csv) {
            csv << s.m << ',' << s.n << ',' << s.k << ','
                << fp32_gemm::kernel_name(dispatch.kind) << ',' << custom_ms << ','
                << custom_gflops << ',' << cublas_ms << ',' << cublas_gflops << ','
                << ratio << ',' << (valid ? 1 : 0) << ',' << max_abs << ',' << max_rel << '\n';
        }
    }

    const double mean = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
    double variance = 0.0;
    for (const double ratio : ratios) {
        variance += (ratio - mean) * (ratio - mean);
    }
    variance /= ratios.size();
    std::cout << "Summary: mean cuBLAS ratio = " << std::fixed << std::setprecision(2) << mean
              << "%, stddev = " << std::sqrt(variance) << " percentage points, correctness = "
              << (all_valid ? "PASS" : "FAIL") << '\n';

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
