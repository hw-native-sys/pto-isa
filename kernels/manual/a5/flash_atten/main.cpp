/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*
 * Standalone driver for TFA (no gtest)
 */

#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <set>

#include "test_common.h"
#include "runtime/rt.h"
#include "fa_performance_kernel.h"
#include "generated_cases.h"

using namespace std;
using namespace PtoTestCommon;

#define GOP_PRECISION 6
#define TIME_PRECISION 3

static std::vector<std::string> Split(const std::string &s, char delim)
{
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(s);
    while (std::getline(ss, item, delim)) {
        if (!item.empty())
            out.push_back(item);
    }
    return out;
}

static std::string Trim(const std::string &s)
{
    const auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos)
        return "";
    const auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> SplitAny(const std::string &s, const std::string &delims)
{
    std::vector<std::string> out;
    std::string token;
    for (char ch : s) {
        if (delims.find(ch) != std::string::npos) {
            if (!token.empty()) {
                out.push_back(token);
                token.clear();
            }
        } else {
            token.push_back(ch);
        }
    }
    if (!token.empty())
        out.push_back(token);
    return out;
}

// Track current case name for golden IO (replaces gtest naming)
static thread_local std::string g_case_name;
static bool g_enable_intermediate = false;
static thread_local std::string g_fifo_summary;
static int g_chip_id = 0; // device id selected via CLI
static const std::string kReportCsv = "./report.csv";
static double g_sys_cnt_multiple = 20.0; // Default A2/A3

static void AppendReportRow(const std::string &case_name, int head, int s0, int s1, int cube_s0, int cube_s1,
                            int tile_s1, uint64_t start_time, uint64_t end_time, double duration_us,
                            double avg_block_us, double gops, const std::string &tflops_str, bool ok)
{
    const bool exists = std::ifstream(kReportCsv).good();
    std::ofstream ofs(kReportCsv, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[WARN] Unable to open report file: " << kReportCsv << std::endl;
        return;
    }
    if (!exists) {
        ofs << "case,HEAD,S0,S1,CUBE_S0,CUBE_S1,TILE_S1,start_time,end_time,duration_us,avg_block_us,GOPS,TFLOPS,"
               "result\n";
    }
    ofs << case_name << ',' << head << ',' << s0 << ',' << s1 << ',' << cube_s0 << ',' << cube_s1 << ',' << tile_s1
        << ',' << start_time << ',' << end_time << ',' << std::fixed << std::setprecision(TIME_PRECISION) << duration_us
        << ',' << std::setprecision(TIME_PRECISION) << avg_block_us << ',' << std::setprecision(GOP_PRECISION) << gops
        << ',' << tflops_str << ',' << (ok ? "OK" : "NOK") << '\n';
}

std::string GetGoldenDir()
{
    return "./" + g_case_name;
}

/*
 * Template usage:
 * - The template parameter `INTERMEDIATE_CHECK` (default false) enables
 *   extra, more-detailed intermediate-value checks. When enabled, the
 *   host will compare the device softmax/intermediate tensor outputs
 *   (e.g. `p_out` / xexp) against golden files. On the device side the
 *   kernel should perform the necessary TSTORE operations to expose
 *   these intermediate buffers for host readback.
 *
 * Example:
 *   run_tfa<float, 64, 128, 256, true>(); // enable intermediate checks
 */
template <typename T, int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD,
          bool INTERMEDIATE_CHECK, bool CAUSAL_MASK>
void run_tfa()
{
    constexpr int tile_factor = TILE_S1 / CUBE_S1;
    constexpr size_t qk_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0) *
                                      static_cast<size_t>(tile_factor) * static_cast<size_t>(CUBE_S1);
    constexpr size_t p_fifo_stride = qk_fifo_stride;
    constexpr size_t p_max_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0);
    constexpr size_t pv_fifo_stride =
        static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0) * static_cast<size_t>(HEAD_SIZE);
    const size_t block_rows = S0 / CUBE_S0;
    constexpr size_t cv_comm_slots = static_cast<size_t>(S0) / static_cast<size_t>(CUBE_S0);
    constexpr size_t cv_comm_bytes = cv_comm_slots * kFaCvCommSlotBytes;
    constexpr size_t profile_bytes_per_block = kFaProfileBytesPerBlock; // cube + two vec subblocks
    const size_t profile_bytes = profile_bytes_per_block * block_rows;

    g_fifo_summary.clear();

    size_t qSize = S0 * HEAD_SIZE * sizeof(aclFloat16);
    size_t kSize = HEAD_SIZE * S1 * sizeof(aclFloat16);

    const size_t qk_fifo_bytes = qk_fifo_stride * block_rows * sizeof(T);
    const size_t p_fifo_bytes_half = p_fifo_stride * block_rows * sizeof(aclFloat16);
    const size_t p_fifo_bytes_float = p_max_fifo_stride * block_rows * sizeof(float);
    const size_t pv_fifo_bytes = pv_fifo_stride * block_rows * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(g_chip_id);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *outHost;
    aclFloat16 *qHost, *kHost;
    aclFloat16 *xexpHost;
    float *tmpFloatExpHost;
    aclFloat16 *vHost;
    T *outDevice; // qk_out
    aclFloat16 *xexpDevice;
    T *midDevice = nullptr; // not used by this test but kept for symmetry
    aclFloat16 *qDevice, *kDevice;
    aclFloat16 *vDevice;
    T *out2Device; // pv_out
    T *out2Host;

    aclrtMallocHost((void **)(&outHost), qk_fifo_bytes); // Allocate qk FIFO buffer
    aclrtMallocHost((void **)(&qHost), qSize);
    aclrtMallocHost((void **)(&kHost), kSize);

    aclrtMalloc((void **)&outDevice, qk_fifo_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&qDevice, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&kDevice, kSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&xexpDevice, p_fifo_bytes_half, ACL_MEM_MALLOC_HUGE_FIRST); // p_out (half) FIFO layout
    void *expMaxIfifoDevice = nullptr;
    aclrtMalloc((void **)&expMaxIfifoDevice, p_fifo_bytes_float,
                ACL_MEM_MALLOC_HUGE_FIRST); // exp_max ififo (float) FIFO layout
    uint8_t *profileDevice = nullptr;
    aclrtMalloc((void **)&profileDevice, profile_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    uint8_t *cvCommDevice = nullptr;
    aclrtMalloc((void **)&cvCommDevice, cv_comm_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate v and out2 buffers
    size_t vSize = S1 * HEAD_SIZE * sizeof(aclFloat16);
    size_t pvPartSize = S0 * HEAD_SIZE * sizeof(T);
    int num_tiles = S1 / TILE_S1;
    size_t out2TotalSize = pv_fifo_bytes;
    aclrtMalloc((void **)&vDevice, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&out2Device, out2TotalSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate global_sum buffer (per-tile S0 floats)
    size_t gsumTotalElems = static_cast<size_t>(S0) * static_cast<size_t>(num_tiles);
    size_t gsumSize = gsumTotalElems * sizeof(float);
    float *gSumDevice = nullptr;
    aclrtMalloc((void **)&gSumDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate per-tile exp_max buffer (per-tile S0 floats)
    float *expMaxDevice = nullptr;
    aclrtMalloc((void **)&expMaxDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate running output o (S0 x HEAD_SIZE)
    T *oDevice = nullptr;
    size_t oSize = pvPartSize; // S0 * HEAD_SIZE * sizeof(T)
    aclrtMalloc((void **)&oDevice, oSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // allocate per-iteration running output snapshots (num_tiles * S0 * HEAD_SIZE)
    T *oPartsDevice = nullptr;
    size_t oPartsTotalSize = pvPartSize * num_tiles;
    aclrtMalloc((void **)&oPartsDevice, oPartsTotalSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // write device buffer addresses/sizes for debug
    auto write_dev_entry = [](std::ofstream &ofs, const std::string &name, uint64_t addr, size_t bytes) {
        ofs << "[" << name << "]\n";
        ofs << "addr = \"0x" << std::hex << addr << "\"\n";
        ofs << std::dec;
        ofs << "size_bytes = " << bytes << "\n\n";
    };
    std::ofstream devToml("./device_addrs.toml", std::ios::out | std::ios::trunc);
    if (devToml.is_open()) {
        write_dev_entry(devToml, "q_device", reinterpret_cast<uint64_t>(qDevice), qSize);
        write_dev_entry(devToml, "k_device", reinterpret_cast<uint64_t>(kDevice), kSize);
        write_dev_entry(devToml, "v_device", reinterpret_cast<uint64_t>(vDevice), vSize);
        write_dev_entry(devToml, "qk_tile_fifo", reinterpret_cast<uint64_t>(outDevice), qk_fifo_bytes);
        write_dev_entry(devToml, "pv_tile_fifo", reinterpret_cast<uint64_t>(out2Device), out2TotalSize);
        write_dev_entry(devToml, "p_tile_fifo", reinterpret_cast<uint64_t>(xexpDevice), p_fifo_bytes_half);
        write_dev_entry(devToml, "exp_max_ififo", reinterpret_cast<uint64_t>(expMaxIfifoDevice), p_fifo_bytes_float);
        write_dev_entry(devToml, "o_out", reinterpret_cast<uint64_t>(oDevice), oSize);
        devToml.close();
    }

    ReadFile(GetGoldenDir() + "/q.bin", qSize, qHost, qSize); // Read q data
    ReadFile(GetGoldenDir() + "/kt.bin", kSize, kHost, kSize);
    // read v
    aclrtMallocHost((void **)(&vHost), S1 * HEAD_SIZE * sizeof(aclFloat16));
    ReadFile(GetGoldenDir() + "/v.bin", vSize, vHost, vSize);

    aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Debug logging setup (preserve original tqksv behavior)
    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    // logging disabled

    if constexpr (INTERMEDIATE_CHECK) {
        std::cout << "[INFO] Intermediate checking is enabled" << std::endl;
    } else {
        std::cout << "[INFO] Intermediate checking is disabled" << std::endl;
    }

    // Launch kernel, pass ffts ctrl addr and device-side log buffer, and xexp/tmp_float_exp device ptrs
    LaunchTFA<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, INTERMEDIATE_CHECK, CAUSAL_MASK,
              kFaCvFifoConsSyncPeriod>(
        (uint16_t *)ffts, (aclFloat16 *)qDevice, (aclFloat16 *)kDevice, (aclFloat16 *)vDevice, (aclFloat16 *)xexpDevice,
        (float *)expMaxIfifoDevice, (float *)gSumDevice, (float *)expMaxDevice, (float *)oDevice, (float *)oPartsDevice,
        (float *)outDevice, (float *)out2Device, profileDevice, stream, cvCommDevice);

    aclrtSynchronizeStream(stream);

    // copy outputs back
    aclrtMemcpy(outHost, qk_fifo_bytes, outDevice, qk_fifo_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMallocHost((void **)(&xexpHost), p_fifo_bytes_half);
    aclrtMallocHost((void **)(&tmpFloatExpHost), p_fifo_bytes_float);
    aclrtMemcpy(xexpHost, p_fifo_bytes_half, xexpDevice, p_fifo_bytes_half, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(tmpFloatExpHost, p_fifo_bytes_float, expMaxIfifoDevice, p_fifo_bytes_float, ACL_MEMCPY_DEVICE_TO_HOST);
    // copy second matmul partial outputs (FIFO layout)
    aclrtMallocHost((void **)(&out2Host), out2TotalSize);
    aclrtMemcpy(out2Host, out2TotalSize, out2Device, out2TotalSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy profiling data back
    uint8_t *profileHost = nullptr;
    aclrtMallocHost((void **)(&profileHost), profile_bytes);
    aclrtMemcpy(profileHost, profile_bytes, profileDevice, profile_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy global_sum back
    float *gSumHost = nullptr;
    aclrtMallocHost((void **)(&gSumHost), gsumSize);
    aclrtMemcpy(gSumHost, gsumSize, gSumDevice, gsumSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy exp_max back
    float *expMaxHost = nullptr;
    aclrtMallocHost((void **)(&expMaxHost), gsumSize);
    aclrtMemcpy(expMaxHost, gsumSize, expMaxDevice, gsumSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy running output o back
    T *oHost = nullptr;
    aclrtMallocHost((void **)(&oHost), oSize);
    aclrtMemcpy(oHost, oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // copy per-iteration o parts back
    T *oPartsHost = nullptr;
    aclrtMallocHost((void **)(&oPartsHost), oPartsTotalSize);
    aclrtMemcpy(oPartsHost, oPartsTotalSize, oPartsDevice, oPartsTotalSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/qk_out.bin", outHost, qk_fifo_bytes);
    WriteFile(GetGoldenDir() + "/p_out.bin", xexpHost, p_fifo_bytes_half);
    WriteFile(GetGoldenDir() + "/exp_max_ififo.bin", tmpFloatExpHost, p_fifo_bytes_float);
    WriteFile(GetGoldenDir() + "/out2.bin", out2Host, out2TotalSize);

    if constexpr (INTERMEDIATE_CHECK) {
        const size_t qk_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0) *
                                      static_cast<size_t>(tile_factor) * static_cast<size_t>(CUBE_S1);
        const size_t p_fifo_stride = qk_fifo_stride; // same dimensions as qk (Cube_S0 x kTileFactor x Cube_S1)
        const size_t p_max_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0);
        const size_t pv_fifo_stride =
            static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0) * static_cast<size_t>(HEAD_SIZE);
        const int block_rows = S0 / CUBE_S0;
        const int fifo_start_tile = std::max(0, num_tiles - kFaCvFifoSize);
        for (int b = 0; b < block_rows; ++b) {
            const size_t qk_off = static_cast<size_t>(b) * qk_fifo_stride;
            const size_t p_off = static_cast<size_t>(b) * p_fifo_stride;
            const size_t p_max_off = static_cast<size_t>(b) * p_max_fifo_stride;
            const size_t pv_off = static_cast<size_t>(b) * pv_fifo_stride;
            WriteFile(GetGoldenDir() + "/block" + std::to_string(b) + "_qk_fifo.bin", outHost + qk_off,
                      qk_fifo_stride * sizeof(float));
            WriteFile(GetGoldenDir() + "/block" + std::to_string(b) + "_p_fifo.bin",
                      reinterpret_cast<uint8_t *>(xexpHost) + p_off * sizeof(aclFloat16),
                      p_fifo_stride * sizeof(aclFloat16));
            WriteFile(GetGoldenDir() + "/block" + std::to_string(b) + "_p_max_fifo.bin",
                      reinterpret_cast<uint8_t *>(tmpFloatExpHost) + p_max_off * sizeof(float),
                      p_max_fifo_stride * sizeof(float));
            WriteFile(GetGoldenDir() + "/block" + std::to_string(b) + "_pv_fifo.bin",
                      reinterpret_cast<uint8_t *>(out2Host) + pv_off * sizeof(float), pv_fifo_stride * sizeof(float));
        }
    }
    // write per-tile global_sum parts
    for (int ti = 0; ti < num_tiles; ++ti) {
        size_t partOffset = static_cast<size_t>(ti) * static_cast<size_t>(S0);
        WriteFile(GetGoldenDir() + "/global_sum_part" + std::to_string(ti) + "_out.bin", gSumHost + partOffset,
                  S0 * sizeof(float));
    }
    // write per-tile exp_max parts
    for (int ti = 0; ti < num_tiles; ++ti) {
        size_t partOffset = static_cast<size_t>(ti) * static_cast<size_t>(S0);
        WriteFile(GetGoldenDir() + "/exp_max_part" + std::to_string(ti) + "_out.bin", expMaxHost + partOffset,
                  S0 * sizeof(float));
    }
    // write running output
    WriteFile(GetGoldenDir() + "/o_out.bin", oHost, oSize);
    // write per-iteration running output snapshots
    for (int ti = 0; ti < num_tiles; ++ti) {
        size_t byteOffset = static_cast<size_t>(ti) * pvPartSize;
        WriteFile(GetGoldenDir() + "/o_part" + std::to_string(ti) + "_out.bin", ((uint8_t *)oPartsHost) + byteOffset,
                  pvPartSize);
    }

    if constexpr (INTERMEDIATE_CHECK) {
        // Build expected FIFO contents from golden tensors and compare against device dumps
        const int block_rows = S0 / CUBE_S0;
        const size_t qk_fifo_stride =
            static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0) * static_cast<size_t>(TILE_S1);
        const size_t p_fifo_stride = qk_fifo_stride; // same dims as qk (Cube_S0 x TILE_S1)
        const size_t p_max_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0);
        const size_t pv_fifo_stride =
            static_cast<size_t>(kFaCvFifoSize) * static_cast<size_t>(CUBE_S0) * static_cast<size_t>(HEAD_SIZE);
        const int fifo_start_tile = std::max(0, num_tiles - kFaCvFifoSize);

        std::vector<float> golden_qk(S0 * S1);
        size_t qk_file_size = 0;
        ReadFile(GetGoldenDir() + "/qk.bin", qk_file_size, golden_qk.data(), golden_qk.size() * sizeof(float));

        std::vector<aclFloat16> golden_p_half(S0 * S1);
        size_t p_file_size = 0;
        ReadFile(GetGoldenDir() + "/p.bin", p_file_size, golden_p_half.data(),
                 golden_p_half.size() * sizeof(aclFloat16));

        std::vector<float> golden_p(golden_p_half.size());
        for (size_t i = 0; i < golden_p_half.size(); ++i) {
            golden_p[i] = aclFloat16ToFloat(golden_p_half[i]);
        }

        std::vector<std::vector<float>> golden_pv_tiles(num_tiles,
                                                        std::vector<float>(static_cast<size_t>(S0) * HEAD_SIZE));
        for (int ti = 0; ti < num_tiles; ++ti) {
            std::string fname = GetGoldenDir() + "/pv_tile_fifo" + std::to_string(ti) + ".bin";
            size_t pv_file_size = 0;
            ReadFile(fname, pv_file_size, golden_pv_tiles[ti].data(), golden_pv_tiles[ti].size() * sizeof(float));
        }

        std::vector<std::vector<float>> golden_exp_max_tiles(num_tiles, std::vector<float>(static_cast<size_t>(S0)));
        for (int ti = 0; ti < num_tiles; ++ti) {
            std::string fname = GetGoldenDir() + "/exp_max_part" + std::to_string(ti) + ".bin";
            size_t exp_max_file_size = 0;
            ReadFile(fname, exp_max_file_size, golden_exp_max_tiles[ti].data(),
                     golden_exp_max_tiles[ti].size() * sizeof(float));
        }

        auto cmp_buf = [](const float *ref, const float *got, size_t count, const std::string &label) {
            std::vector<float> ref_vec(ref, ref + count);
            std::vector<float> got_vec(got, got + count);
            const bool is_exp_max =
                (label.find("p_max") != std::string::npos) || (label.find("exp_max") != std::string::npos);
            const float tol = is_exp_max ? 1e-2f : 1e-3f;
            std::cout << "[CHECK] comparing " << label << " count=" << count << " tol=" << tol << std::endl;
            const bool ok = ResultCmp<float>(ref_vec, got_vec, tol);
            if (!ok) {
                std::cerr << "[INTERMEDIATE MISMATCH] " << label << std::endl;
            }
            return ok;
        };

        std::set<int> fail_qk_tiles;
        std::set<int> fail_p_tiles;
        std::set<int> fail_p_max_tiles;
        std::set<int> fail_pv_tiles;
        bool all_ok = true;
        for (int b = 0; b < block_rows; ++b) {
            bool block_qk_ok = true;
            bool block_p_ok = true;
            bool block_p_max_ok = true;
            bool block_pv_ok = true;
            // Expected FIFOs
            std::vector<float> exp_qk(qk_fifo_stride, 0.0f);
            std::vector<float> exp_p(p_fifo_stride, 0.0f);
            std::vector<float> exp_p_max(p_max_fifo_stride, 0.0f);
            std::vector<float> exp_pv(pv_fifo_stride, 0.0f);

            for (int ti = fifo_start_tile; ti < num_tiles; ++ti) {
                const uint32_t buf_idx = static_cast<uint32_t>(ti % kFaCvFifoSize);
                size_t qk_off = static_cast<size_t>(buf_idx) * static_cast<size_t>(CUBE_S0) *
                                static_cast<size_t>(tile_factor) * static_cast<size_t>(CUBE_S1);
                size_t p_off = qk_off;
                size_t p_max_off = static_cast<size_t>(buf_idx) * static_cast<size_t>(CUBE_S0);
                size_t pv_off =
                    static_cast<size_t>(buf_idx) * static_cast<size_t>(CUBE_S0) * static_cast<size_t>(HEAD_SIZE);

                const int c0 = ti * TILE_S1;

                // DN builds dump qk/p subtiles as (Cube_S1 x Cube_S0); ND builds dump them as (Cube_S0 x Cube_S1).
#ifdef MODE_DN
                for (int sub_col = 0; sub_col < tile_factor; ++sub_col) {
                    const size_t subtile_off = qk_off + static_cast<size_t>(sub_col) * static_cast<size_t>(CUBE_S0) *
                                                            static_cast<size_t>(CUBE_S1);
                    for (int c = 0; c < CUBE_S1; ++c) {
                        float *qk_dst = &exp_qk[subtile_off + static_cast<size_t>(c) * static_cast<size_t>(CUBE_S0)];
                        float *p_dst = &exp_p[subtile_off + static_cast<size_t>(c) * static_cast<size_t>(CUBE_S0)];
                        for (int r = 0; r < CUBE_S0; ++r) {
                            const int global_r = b * CUBE_S0 + r;
                            const size_t src_idx =
                                static_cast<size_t>(global_r) * static_cast<size_t>(S1) + c0 + sub_col * CUBE_S1 + c;
                            qk_dst[r] = golden_qk[src_idx];
                            p_dst[r] = golden_p[src_idx];
                        }
                    }
                }
#else
                // Copy qk tile
                for (int r = 0; r < CUBE_S0; ++r) {
                    const int global_r = b * CUBE_S0 + r;
                    for (int sub_col = 0; sub_col < tile_factor; ++sub_col) {
                        const float *src = &golden_qk[static_cast<size_t>(global_r) * S1 + c0 + sub_col * CUBE_S1];
                        float *dst = &exp_qk[qk_off +
                                             static_cast<size_t>(sub_col) * static_cast<size_t>(CUBE_S0) *
                                                 static_cast<size_t>(CUBE_S1) +
                                             static_cast<size_t>(r) * CUBE_S1];
                        std::copy_n(src, CUBE_S1, dst);
                    }
                }

                // Copy p tile (converted to float) with new layout: contiguous sub-tiles of width CUBE_S1
                for (int r = 0; r < CUBE_S0; ++r) {
                    const int global_r = b * CUBE_S0 + r;
                    for (int sub_col = 0; sub_col < tile_factor; ++sub_col) {
                        const float *src = &golden_p[static_cast<size_t>(global_r) * S1 + c0 + sub_col * CUBE_S1];
                        float *dst = &exp_p[p_off +
                                            static_cast<size_t>(sub_col) * static_cast<size_t>(CUBE_S0) *
                                                static_cast<size_t>(CUBE_S1) +
                                            static_cast<size_t>(r) * CUBE_S1];
                        std::copy_n(src, CUBE_S1, dst);
                    }
                }
#endif

                for (int r = 0; r < CUBE_S0; ++r) {
                    const int global_r = b * CUBE_S0 + r;
                    exp_p_max[p_max_off + static_cast<size_t>(r)] = golden_exp_max_tiles[ti][global_r];
                }

                const std::vector<float> &pv_tile = golden_pv_tiles[ti];
                for (int r = 0; r < CUBE_S0; ++r) {
                    const int global_r = b * CUBE_S0 + r;
                    const float *src = &pv_tile[static_cast<size_t>(global_r) * HEAD_SIZE];
                    float *dst = &exp_pv[pv_off + static_cast<size_t>(r) * HEAD_SIZE];
                    std::copy_n(src, HEAD_SIZE, dst);
                }
            }

            // Load device dump for this block
            std::vector<float> got_qk(qk_fifo_stride);
            std::vector<aclFloat16> got_p_half(p_fifo_stride);
            std::vector<float> got_p(p_fifo_stride);
            std::vector<float> got_p_max(p_max_fifo_stride);
            std::vector<float> got_pv(pv_fifo_stride);

            size_t qk_block_file_size = 0;
            ReadFile(GetGoldenDir() + "/block" + std::to_string(b) + "_qk_fifo.bin", qk_block_file_size, got_qk.data(),
                     got_qk.size() * sizeof(float));
            size_t p_block_file_size = 0;
            ReadFile(GetGoldenDir() + "/block" + std::to_string(b) + "_p_fifo.bin", p_block_file_size,
                     got_p_half.data(), got_p_half.size() * sizeof(aclFloat16));
            size_t p_max_block_file_size = 0;
            ReadFile(GetGoldenDir() + "/block" + std::to_string(b) + "_p_max_fifo.bin", p_max_block_file_size,
                     got_p_max.data(), got_p_max.size() * sizeof(float));
            for (size_t i = 0; i < got_p_half.size(); ++i) {
                got_p[i] = aclFloat16ToFloat(got_p_half[i]);
            }
            size_t pv_block_file_size = 0;
            ReadFile(GetGoldenDir() + "/block" + std::to_string(b) + "_pv_fifo.bin", pv_block_file_size, got_pv.data(),
                     got_pv.size() * sizeof(float));

            for (int ti = fifo_start_tile; ti < num_tiles; ++ti) {
                const uint32_t s0_index = b * CUBE_S0;
                const uint32_t s1_index = ti * TILE_S1;
                const bool skip_for_causal_mask = CAUSAL_MASK && (s1_index > s0_index);

                const uint32_t buf_idx = static_cast<uint32_t>(ti % kFaCvFifoSize);
                const size_t qk_off = static_cast<size_t>(buf_idx) * static_cast<size_t>(CUBE_S0) *
                                      static_cast<size_t>(tile_factor) * static_cast<size_t>(CUBE_S1);
                const size_t p_off = qk_off;
                const size_t p_max_off = static_cast<size_t>(buf_idx) * static_cast<size_t>(CUBE_S0);
                const size_t pv_off =
                    static_cast<size_t>(buf_idx) * static_cast<size_t>(CUBE_S0) * static_cast<size_t>(HEAD_SIZE);

                const size_t qk_tile_elems =
                    static_cast<size_t>(CUBE_S0) * static_cast<size_t>(tile_factor) * static_cast<size_t>(CUBE_S1);
                const size_t p_tile_elems = static_cast<size_t>(CUBE_S0) * static_cast<size_t>(TILE_S1);
                const size_t pv_tile_elems = static_cast<size_t>(CUBE_S0) * static_cast<size_t>(HEAD_SIZE);

                const std::string blk_tile = " block " + std::to_string(b) + " tile " + std::to_string(ti);
                const bool tile_qk_ok = skip_for_causal_mask ? true :
                                                               cmp_buf(&exp_qk[qk_off], &got_qk[qk_off], qk_tile_elems,
                                                                       "qk_fifo" + blk_tile);
                const bool tile_p_ok = skip_for_causal_mask ?
                                           true :
                                           cmp_buf(&exp_p[p_off], &got_p[p_off], p_tile_elems, "p_fifo" + blk_tile);
                block_qk_ok = block_qk_ok && tile_qk_ok;
                block_p_ok = block_p_ok && tile_p_ok;
                if (!tile_qk_ok)
                    fail_qk_tiles.insert(ti);
                if (!tile_p_ok)
                    fail_p_tiles.insert(ti);
                const bool tile_pv_ok = skip_for_causal_mask ? true :
                                                               cmp_buf(&exp_pv[pv_off], &got_pv[pv_off], pv_tile_elems,
                                                                       "pv_fifo" + blk_tile);
                block_pv_ok = block_pv_ok && tile_pv_ok;
                if (!tile_pv_ok)
                    fail_pv_tiles.insert(ti);
                // exp_max fifo is 1D per row; tile 0 is skipped
                if (ti != 0) {
                    std::vector<float> exp_p_max_row(CUBE_S0);
                    std::vector<float> got_p_max_row(CUBE_S0);
                    for (int r = 0; r < CUBE_S0; ++r) {
                        exp_p_max_row[r] = exp_p_max[p_max_off + static_cast<size_t>(r)];
                        got_p_max_row[r] = got_p_max[p_max_off + static_cast<size_t>(r)];
                    }
                    const bool tile_p_max_ok = skip_for_causal_mask ?
                                                   true :
                                                   cmp_buf(exp_p_max_row.data(), got_p_max_row.data(),
                                                           exp_p_max_row.size(), "p_max_fifo" + blk_tile);
                    block_p_max_ok = block_p_max_ok && tile_p_max_ok;
                    if (!tile_p_max_ok)
                        fail_p_max_tiles.insert(ti);
                }
            }
            const bool block_ok = block_qk_ok && block_p_ok && block_p_max_ok && block_pv_ok;
            std::cout << "[CHECK] block " << b << " qk=" << (block_qk_ok ? "OK" : "FAIL")
                      << " p=" << (block_p_ok ? "OK" : "FAIL") << " p_max=" << (block_p_max_ok ? "OK" : "FAIL")
                      << " pv=" << (block_pv_ok ? "OK" : "FAIL") << std::endl;
            all_ok = all_ok && block_ok;
        }

        auto print_fail_summary = [](const std::string &label, const std::set<int> &fails) {
            if (fails.empty()) {
                std::cout << "[CHECK] " << label << " tiles: all OK" << std::endl;
            } else {
                std::cout << "[CHECK] " << label << " tiles failed: ";
                bool first = true;
                for (int ti : fails) {
                    if (!first)
                        std::cout << ",";
                    std::cout << ti;
                    first = false;
                }
                std::cout << std::endl;
            }
        };

        print_fail_summary("qk_fifo", fail_qk_tiles);
        print_fail_summary("p_fifo", fail_p_tiles);
        print_fail_summary("p_max_fifo", fail_p_max_tiles);
        print_fail_summary("pv_fifo", fail_pv_tiles);

        auto set_to_string = [](const std::set<int> &s) {
            std::string out;
            bool first = true;
            for (int v : s) {
                if (!first)
                    out += ",";
                out += std::to_string(v);
                first = false;
            }
            return out.empty() ? std::string("-") : out;
        };

        g_fifo_summary = "[SUMMARY] fifo fails -> qk:" + set_to_string(fail_qk_tiles) +
                         " p:" + set_to_string(fail_p_tiles) + " p_max:" + set_to_string(fail_p_max_tiles) +
                         " pv:" + set_to_string(fail_pv_tiles);

        std::cout << (all_ok ? "[CHECK] FIFO intermediate ok" : "[CHECK] FIFO intermediate FAILED") << std::endl;
    } else {
        std::cout << "[INFO] Intermediate checking skipped; proceeding to final compare only" << std::endl;
    }

    aclrtFree(outDevice);
    aclrtFree(oDevice);
    aclrtFree(oPartsDevice);
    aclrtFree(qDevice);
    aclrtFree(kDevice);
    aclrtFree(xexpDevice);
    aclrtFree(expMaxIfifoDevice);
    aclrtFree(vDevice);
    aclrtFree(out2Device);
    aclrtFree(gSumDevice);
    aclrtFree(expMaxDevice);
    aclrtFree(profileDevice);
    aclrtFree(cvCommDevice);

    // Final running output compare
    std::vector<float> golden_o(S0 * HEAD_SIZE);
    std::vector<float> dev_o(S0 * HEAD_SIZE);
    size_t golden_o_file_size = 0;
    ReadFile(GetGoldenDir() + "/o.bin", golden_o_file_size, golden_o.data(), oSize);
    size_t dev_o_file_size = 0;
    ReadFile(GetGoldenDir() + "/o_out.bin", dev_o_file_size, dev_o.data(), oSize);
    std::cout << "[CHECK] O running output compare" << std::endl;
    bool o_ok = ResultCmp<float>(golden_o, dev_o, 0.001f);

    uint64_t start_min = std::numeric_limits<uint64_t>::max();
    uint64_t end_max = 0;
    double block_duration_us_sum = 0.0;
    int block_duration_count = 0;
    const size_t profiles_per_block = profile_bytes_per_block / 1024; // 3 (cube + vec subblocks)
    for (size_t b = 0; b < block_rows; ++b) {
        uint64_t block_start = std::numeric_limits<uint64_t>::max();
        uint64_t block_end = 0;
        for (size_t p = 0; p < profiles_per_block; ++p) {
            const uint64_t *entry = reinterpret_cast<uint64_t *>(profileHost + b * profile_bytes_per_block + p * 1024);
            const uint64_t st = entry[0];
            const uint64_t ed = entry[1];
            if (st == 0 && ed == 0) {
                continue;
            }
            block_start = std::min(block_start, st);
            block_end = std::max(block_end, ed);
            start_min = std::min(start_min, st);
            end_max = std::max(end_max, ed);
        }
        if (block_start != std::numeric_limits<uint64_t>::max() && block_end >= block_start) {
            uint64_t block_ticks = block_end - block_start;
            double block_us = static_cast<double>(block_ticks) * g_sys_cnt_multiple / 1000.0;
            block_duration_us_sum += block_us;
            block_duration_count += 1;
        }
    }
    if (start_min == std::numeric_limits<uint64_t>::max()) {
        start_min = 0;
    }
    bool valid_times = (start_min != 0 || end_max != 0) && (end_max >= start_min);
    uint64_t duration_ticks = valid_times ? (end_max - start_min) : 0;
    double duration_ns = static_cast<double>(duration_ticks) * g_sys_cnt_multiple;
    double duration_us = duration_ns / 1000.0;
    double gops = static_cast<double>(S0) * static_cast<double>(S1) * static_cast<double>(HEAD_SIZE) * 4.0 / 1e6;
    std::string tflops_str;
    if (!valid_times) {
        tflops_str = "NA";
    } else {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(GOP_PRECISION) << gops / (duration_us + 1e-9);
        tflops_str = oss.str();
    }
    double avg_block_us = 0.0;
    if (block_duration_count > 0) {
        avg_block_us = block_duration_us_sum / static_cast<double>(block_duration_count);
    }
    AppendReportRow(g_case_name, HEAD_SIZE, S0, S1, CUBE_S0, CUBE_S1, TILE_S1, start_min, end_max, duration_us,
                    avg_block_us, gops, tflops_str, o_ok);

    std::cout << (o_ok ? "test success" : "test failed") << std::endl;
    if (!g_fifo_summary.empty()) {
        std::cout << g_fifo_summary << std::endl;
    }
    std::cout << "[SUMMARY] o_out status: " << (o_ok ? "OK" : "FAIL") << std::endl;

    aclrtFreeHost(outHost); // Free host memory
    aclrtFreeHost(qHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(xexpHost);
    aclrtFreeHost(tmpFloatExpHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(out2Host);
    aclrtFreeHost(oHost);
    aclrtFreeHost(oPartsHost);
    aclrtFreeHost(gSumHost);
    aclrtFreeHost(expMaxHost);
    aclrtFreeHost(profileHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(g_chip_id);
    aclFinalize();
}

template <typename T, int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD,
          bool CAUSAL_MASK>
void run_case(const std::string &case_name)
{
    g_case_name = case_name;
    if (g_enable_intermediate) {
        run_tfa<T, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, true, CAUSAL_MASK>();
    } else {
        run_tfa<T, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, false, CAUSAL_MASK>();
    }
}

int main(int argc, char **argv)
{
    struct CaseEntry {
        std::string name;
        std::function<void()> run;
    };

    std::vector<CaseEntry> cases = {
#define TFA_CASE_ENTRY(S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CAUSAL_MASK)                           \
    {"case_float_H_" #HEAD "_S0_" #S0 "_S1_" #S1, []() {                                                           \
         run_case<float, S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CAUSAL_MASK>("case_float_H_" #HEAD   \
                                                                                           "_S0_" #S0 "_S1_" #S1); \
     }},
        TFA_FOR_EACH_CASE(TFA_CASE_ENTRY)
#undef TFA_CASE_ENTRY
    };

    std::vector<std::string> filters;
    std::string filter_arg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const std::string prefix_case = "--case=";
        const std::string prefix_cases = "--cases=";
        const std::string prefix_chip = "--chip=";
        const std::string prefix_npu = "--npu=";
        const std::string prefix_sys_cnt_mtp = "--sys_cnt_multiple=";

        if (arg.rfind(prefix_case, 0) == 0) {
            filter_arg = arg.substr(prefix_case.size());
            continue;
        }
        if (arg.rfind(prefix_cases, 0) == 0) {
            filter_arg = arg.substr(prefix_cases.size());
            continue;
        }
        if (arg.rfind(prefix_chip, 0) == 0) {
            g_chip_id = std::stoi(arg.substr(prefix_chip.size()));
            continue;
        }
        if (arg.rfind(prefix_npu, 0) == 0) {
            g_chip_id = std::stoi(arg.substr(prefix_npu.size()));
            continue;
        }
        if (arg.rfind(prefix_sys_cnt_mtp, 0) == 0) {
            g_sys_cnt_multiple = std::atof(arg.substr(prefix_sys_cnt_mtp.size()).c_str());
            continue;
        }
        if ((arg == "--case" || arg == "--cases") && (i + 1) < argc) {
            filter_arg = argv[++i];
            continue;
        }
        if ((arg == "--chip" || arg == "-c") && (i + 1) < argc) {
            g_chip_id = std::stoi(argv[++i]);
            continue;
        }
        if ((arg == "--npu" || arg == "-n") && (i + 1) < argc) {
            g_chip_id = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--intermediate" || arg == "-i" || arg == "-I") {
            g_enable_intermediate = true;
            continue;
        }
        if (arg.rfind("--intermediate=", 0) == 0) {
            std::string val = arg.substr(std::strlen("--intermediate="));
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            g_enable_intermediate = (val == "1" || val == "true" || val == "yes");
            continue;
        }
        if ((arg == "--sys_cnt_multiple") && (i + 1) < argc) {
            g_sys_cnt_multiple = std::atof(argv[++i]);
            continue;
        }
    }
    if (!filter_arg.empty()) {
        // Split multiple cases by ';' only to preserve comma-separated tuple tokens
        std::vector<std::string> raw_filters = Split(filter_arg, ';');
        auto normalize_filter = [](const std::string &f) {
            // Accept either canonical case name or numeric tuple HEAD,S0,S1[,CUBE_S0[,TILE_S1]]
            const std::string trimmed = Trim(f);
            if (trimmed.rfind("case_float", 0) == 0)
                return trimmed;
            std::vector<std::string> parts = Split(trimmed, ',');
            if (parts.size() >= 3) {
                try {
                    int head = std::stoi(Trim(parts[0]));
                    int s0 = std::stoi(Trim(parts[1]));
                    int s1 = std::stoi(Trim(parts[2]));
                    return std::string("case_float_H_") + std::to_string(head) + "_S0_" + std::to_string(s0) + "_S1_" +
                           std::to_string(s1);
                } catch (...) {
                    return trimmed; // fallback to trimmed original
                }
            }
            return trimmed;
        };
        for (auto &f : raw_filters) {
            const std::string norm = normalize_filter(f);
            if (!norm.empty())
                filters.push_back(norm);
        }
    }

    std::cout << "[DEBUG] Available cases (" << cases.size() << "): ";
    for (size_t i = 0; i < cases.size(); ++i) {
        std::cout << cases[i].name;
        if (i + 1 != cases.size())
            std::cout << ",";
    }
    std::cout << std::endl;

    if (!filters.empty()) {
        std::cout << "[DEBUG] Requested filters: ";
        for (size_t i = 0; i < filters.size(); ++i) {
            std::cout << filters[i];
            if (i + 1 != filters.size())
                std::cout << ",";
        }
        std::cout << std::endl;
    } else {
        std::cout << "[DEBUG] No filters provided; running all cases" << std::endl;
    }

    auto should_run = [&](const std::string &name) {
        if (filters.empty())
            return true;
        return std::find(filters.begin(), filters.end(), name) != filters.end();
    };

    std::vector<std::string> to_run;
    for (const auto &c : cases) {
        if (should_run(c.name)) {
            to_run.push_back(c.name);
        }
    }

    if (to_run.empty()) {
        if (!filters.empty()) {
            std::cerr << "[WARN] No cases matched filters; check --case/--cases values." << std::endl;
        } else {
            std::cerr << "[WARN] No cases available to run." << std::endl;
        }
        return 1;
    }

    std::cout << "[DEBUG] Will run cases (" << to_run.size() << "): ";
    for (size_t i = 0; i < to_run.size(); ++i) {
        std::cout << to_run[i];
        if (i + 1 != to_run.size())
            std::cout << ",";
    }
    std::cout << std::endl;

    for (const auto &c : cases) {
        if (should_run(c.name)) {
            c.run();
        }
    }

    return 0;
}
