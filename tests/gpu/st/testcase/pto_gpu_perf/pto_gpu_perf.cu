#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include <pto/pto-inst.hpp>

namespace {

template <typename T>
__global__ void KernelTMATMULBench(float *out, T *a, T *b)
{
    using TileA = pto::Tile<pto::TileType::Vec, T, 64, 64, pto::BLayout::RowMajor, -1, -1>;
    using TileB = pto::Tile<pto::TileType::Vec, T, 64, 64, pto::BLayout::RowMajor, -1, -1>;
    using TileC = pto::Tile<pto::TileType::Acc, float, 64, 64, pto::BLayout::RowMajor, -1, -1>;
    TileA aTile(64, 64);
    TileB bTile(64, 64);
    TileC cTile(64, 64);
    aTile.data() = a;
    bTile.data() = b;
    cTile.data() = out;
    pto::TMATMUL(cTile, aTile, bTile);
}

bool Check(cudaError_t st, const char *what)
{
    if (st == cudaSuccess) return true;
    std::cerr << what << ": " << cudaGetErrorString(st) << std::endl;
    return false;
}

template <typename T>
void InitInputs(std::vector<T> &a, std::vector<T> &b)
{
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<T>((int(i % 17) - 8) * 0.1f);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<T>((int(i % 13) - 6) * 0.2f);
}

template <>
void InitInputs<half>(std::vector<half> &a, std::vector<half> &b)
{
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = __float2half((int(i % 17) - 8) * 0.1f);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = __float2half((int(i % 13) - 6) * 0.2f);
}

template <>
void InitInputs<bfloat16_t>(std::vector<bfloat16_t> &a, std::vector<bfloat16_t> &b)
{
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = __float2bfloat16((int(i % 17) - 8) * 0.1f);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = __float2bfloat16((int(i % 13) - 6) * 0.2f);
}

template <typename T>
bool RunBench(const std::string &name, dim3 block, int iters)
{
    constexpr int M = 64, K = 64, N = 64;
    constexpr std::size_t aCount = M * K;
    constexpr std::size_t bCount = K * N;
    constexpr std::size_t cCount = M * N;

    std::vector<T> hA(aCount), hB(bCount);
    InitInputs(hA, hB);

    T *dA = nullptr;
    T *dB = nullptr;
    float *dC = nullptr;
    if (!Check(cudaMalloc(&dA, aCount * sizeof(T)), "cudaMalloc dA")) return false;
    if (!Check(cudaMalloc(&dB, bCount * sizeof(T)), "cudaMalloc dB")) return false;
    if (!Check(cudaMalloc(&dC, cCount * sizeof(float)), "cudaMalloc dC")) return false;
    if (!Check(cudaMemcpy(dA, hA.data(), aCount * sizeof(T), cudaMemcpyHostToDevice), "copy A")) return false;
    if (!Check(cudaMemcpy(dB, hB.data(), bCount * sizeof(T), cudaMemcpyHostToDevice), "copy B")) return false;
    if (!Check(cudaMemset(dC, 0, cCount * sizeof(float)), "memset C")) return false;

    for (int i = 0; i < 10; ++i) {
        KernelTMATMULBench<T><<<1, block>>>(dC, dA, dB);
    }
    if (!Check(cudaDeviceSynchronize(), "warmup sync")) return false;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        KernelTMATMULBench<T><<<1, block>>>(dC, dA, dB);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double avgMs = ms / iters;
    const double flops = 2.0 * M * N * K;
    const double gflops = flops / (avgMs * 1.0e6);

    std::cout << std::left << std::setw(8) << name
              << " avg_ms=" << std::fixed << std::setprecision(4) << avgMs
              << " gflops=" << std::fixed << std::setprecision(2) << gflops
              << " block=" << block.x << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return true;
}

} // namespace

int main()
{
    bool ok = true;
    std::cout << "PTO GPU matmul microbench on GB10 (64x64x64, 1 block)" << std::endl;
    ok &= RunBench<float>("float", dim3(1), 200);
    ok &= RunBench<half>("half", dim3(32), 500);
    ok &= RunBench<bfloat16_t>("bf16", dim3(32), 500);
    return ok ? 0 : 1;
}
