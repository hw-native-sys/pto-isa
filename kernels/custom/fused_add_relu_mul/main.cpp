/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifdef __CPU_SIM
#include <pto/common/cpu_stub.hpp>
#else
#include "acl/acl.h"
#endif
#include "test_common.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;
using namespace PtoTestCommon;

// Kernel 启动函数声明
template <typename T>
void LaunchFusedAddReLUMul(uint8_t* out, uint8_t* x, float bias, float scale, uint32_t totalLength, void* stream);

template <typename T>
void LaunchFusedAddReLUMulOptimized(
    uint8_t* out, uint8_t* x, float bias, float scale, uint32_t totalLength, void* stream);

template <typename T>
void LaunchFusedAddReLUMulLargeTile(
    uint8_t* out, uint8_t* x, float bias, float scale, uint32_t totalLength, void* stream);

/**
 * @brief CPU 参考实现：计算 golden 结果
 */
void ComputeGolden(float* golden, const float* x, float bias, float scale, uint32_t length)
{
    for (uint32_t i = 0; i < length; i++) {
        // Step 1: Add
        float temp = x[i] + bias;

        // Step 2: ReLU
        temp = (temp > 0.0f) ? temp : 0.0f;

        // Step 3: Mul
        golden[i] = temp * scale;
    }
}

/**
 * @brief 结果比较函数
 */
bool CompareResults(const float* result, const float* golden, uint32_t length, float tolerance = 1e-5f)
{
    float max_diff = 0.0f;
    uint32_t error_count = 0;
    const uint32_t max_errors_to_print = 10;

    for (uint32_t i = 0; i < length; i++) {
        float diff = fabs(result[i] - golden[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }

        if (diff > tolerance) {
            error_count++;
            if (error_count <= max_errors_to_print) {
                printf("  [%u] result=%.6f, golden=%.6f, diff=%.6e\n", i, result[i], golden[i], diff);
            }
        }
    }

    printf("Max difference: %.6e\n", max_diff);
    // 避免除零错误
    if (length > 0) {
        printf("Error count: %u / %u (%.2f%%)\n", error_count, length, 100.0f * error_count / length);
    } else {
        printf("Error count: %u / 0 (N/A)\n", error_count);
    }

    return (max_diff < tolerance);
}

/**
 * @brief 初始化测试数据
 */
static void InitializeTestData(float* x_host, uint32_t length)
{
    // 使用确定性公式生成测试数据，范围在 [-2, 2]
    // 这样可以保证测试的可重复性，同时避免使用不安全的 rand()
    for (uint32_t i = 0; i < length; i++) {
        float normalized = (float)(i % 1000) / 1000.0f; // [0, 1)
        x_host[i] = -2.0f + 4.0f * normalized;
    }
}

/**
 * @brief 分配测试所需的内存资源
 */
static bool AllocateTestMemory(
    uint32_t length, size_t data_size, float** x_host, float** out_host, float** golden_host, uint8_t** x_device,
    uint8_t** out_device)
{
    // 分配 Host 内存
    aclrtMallocHost((void**)x_host, data_size);
    aclrtMallocHost((void**)out_host, data_size);

    // 校验内存分配大小
    if (length > 0 && length <= (1u << 30)) {
        *golden_host = new float[length];
    } else {
        printf("Error: Invalid memory allocation size\n");
        return false;
    }

    // 分配 Device 内存
    aclrtMalloc((void**)x_device, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)out_device, data_size, ACL_MEM_MALLOC_HUGE_FIRST);

    return true;
}

/**
 * @brief 清理测试资源
 */
static void CleanupTestResources(
    float* x_host, float* out_host, float* golden_host, uint8_t* x_device, uint8_t* out_device, aclrtStream stream)
{
    aclrtFree(x_device);
    aclrtFree(out_device);
    aclrtFreeHost(x_host);
    aclrtFreeHost(out_host);
    delete[] golden_host;

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

/**
 * @brief 测试函数模板
 */
template <typename LaunchFunc>
bool TestKernel(const char* kernel_name, LaunchFunc launch_func, uint32_t length, float bias, float scale)
{
    printf("\n========== Testing %s ==========\n", kernel_name);
    printf("Parameters: length=%u, bias=%.2f, scale=%.2f\n", length, bias, scale);

    // 参数校验
    if (length == 0 || length > (1u << 30)) {
        printf("Error: Invalid length parameter: %u\n", length);
        return false;
    }

    size_t data_size = length * sizeof(float);

    // 初始化 ACL
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 分配内存
    float *x_host, *out_host, *golden_host;
    uint8_t *x_device, *out_device;

    if (!AllocateTestMemory(length, data_size, &x_host, &out_host, &golden_host, &x_device, &out_device)) {
        aclrtFreeHost(x_host);
        aclrtFreeHost(out_host);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return false;
    }

    // 初始化输入数据
    InitializeTestData(x_host, length);

    // 计算 CPU golden 结果
    ComputeGolden(golden_host, x_host, bias, scale, length);

    // 拷贝输入数据到 Device
    aclrtMemcpy(x_device, data_size, x_host, data_size, ACL_MEMCPY_HOST_TO_DEVICE);

    // 启动 Kernel
    launch_func((uint8_t*)out_device, (uint8_t*)x_device, bias, scale, length, stream);

    // 同步并拷贝结果回 Host
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(out_host, data_size, out_device, data_size, ACL_MEMCPY_DEVICE_TO_HOST);

    // 比较结果
    bool passed = CompareResults(out_host, golden_host, length);

    // 清理资源
    CleanupTestResources(x_host, out_host, golden_host, x_device, out_device, stream);

    if (passed) {
        printf("✓ %s PASSED\n", kernel_name);
    } else {
        printf("✗ %s FAILED\n", kernel_name);
    }

    return passed;
}

/**
 * @brief 性能测试函数
 */
template <typename LaunchFunc>
void BenchmarkKernel(
    const char* kernel_name, LaunchFunc launch_func, uint32_t length, float bias, float scale, int iterations = 100)
{
    printf("\n========== Benchmarking %s ==========\n", kernel_name);
    printf("Parameters: length=%u, iterations=%d\n", length, iterations);

    size_t data_size = length * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *x_device, *out_device;
    aclrtMalloc((void**)&x_device, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&out_device, data_size, ACL_MEM_MALLOC_HUGE_FIRST);

    // 预热
    for (int i = 0; i < 10; i++) {
        launch_func(out_device, x_device, bias, scale, length, stream);
    }
    aclrtSynchronizeStream(stream);

    // 性能测试
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        launch_func(out_device, x_device, bias, scale, length, stream);
    }
    aclrtSynchronizeStream(stream);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    // 避免除零错误
    if (iterations > 0) {
        double avg_time_ms = elapsed_ms / iterations;
        double throughput_gb_s = (2.0 * data_size / 1e9) / (avg_time_ms / 1000.0); // 读+写

        printf("Average time: %.4f ms\n", avg_time_ms);
        printf("Throughput: %.2f GB/s\n", throughput_gb_s);
    } else {
        printf("Warning: iterations is 0, cannot calculate average time\n");
    }

    aclrtFree(x_device);
    aclrtFree(out_device);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

int main(int argc, char* argv[])
{
    printf("========================================\n");
    printf("Fused Add-ReLU-Mul Custom Operator Test\n");
    printf("========================================\n");

    // 测试参数
    const float bias = 1.0f;
    const float scale = 2.0f;

    bool all_passed = true;

    // ========== 功能测试 ==========
    printf("\n========== Functional Tests ==========\n");

    // 测试1：小规模数据
    all_passed &= TestKernel("Basic Kernel (Small)", LaunchFusedAddReLUMul<float>, 1024, bias, scale);

    // 测试2：中等规模数据
    all_passed &= TestKernel("Basic Kernel (Medium)", LaunchFusedAddReLUMul<float>, 1024 * 1024, bias, scale);

    // 测试3：大规模数据
    all_passed &= TestKernel("Basic Kernel (Large)", LaunchFusedAddReLUMul<float>, 16 * 1024 * 1024, bias, scale);

    // 测试4：优化版本（双缓冲）
    all_passed &=
        TestKernel("Optimized Kernel (Double Buffer)", LaunchFusedAddReLUMulOptimized<float>, 1024 * 1024, bias, scale);

    // 测试5：大 Tile 版本
    all_passed &= TestKernel("Large Tile Kernel", LaunchFusedAddReLUMulLargeTile<float>, 1024 * 1024, bias, scale);

    // ========== 性能测试 ==========
    printf("\n========== Performance Benchmarks ==========\n");

    const uint32_t bench_length = 16 * 1024 * 1024; // 16M 元素 = 64 MB

    BenchmarkKernel("Basic Kernel", LaunchFusedAddReLUMul<float>, bench_length, bias, scale);

    BenchmarkKernel(
        "Optimized Kernel (Double Buffer)", LaunchFusedAddReLUMulOptimized<float>, bench_length, bias, scale);

    BenchmarkKernel("Large Tile Kernel", LaunchFusedAddReLUMulLargeTile<float>, bench_length, bias, scale);

    // ========== 总结 ==========
    printf("\n========================================\n");
    if (all_passed) {
        printf("✓ All tests PASSED\n");
        return 0;
    } else {
        printf("✗ Some tests FAILED\n");
        return 1;
    }
}
