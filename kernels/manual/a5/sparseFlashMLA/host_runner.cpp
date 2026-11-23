/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>

#include "flash_mla_api.h"
#include "flash_mla_common.h"
#include "host_runner.h"

namespace {
using Metadata = std::map<std::string, std::string>;
constexpr size_t CACHE_RECORD_BYTES = flash_mla_bf16::CACHE_RECORD_BYTES;
constexpr const char* CACHE_FORMAT = "deepseek_fp4_288_e4m3_sf16";
constexpr const char* PV_FORMAT = "bf16";

void checkAcl(aclError error, const char* operation)
{
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(error));
    }
}

size_t checkedSize(std::initializer_list<size_t> dimensions)
{
    size_t result = 1;
    for (size_t dimension : dimensions) {
        if (dimension != 0 && result > std::numeric_limits<size_t>::max() / dimension) {
            throw std::runtime_error("Tensor size exceeds addressable memory");
        }
        result *= dimension;
    }
    return result;
}

Metadata readMetadata(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot read metadata: " + path.string());
    }
    Metadata result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 == line.size()) {
            throw std::runtime_error("Invalid fixture.txt line: " + line);
        }
        if (!result.emplace(line.substr(0, equals), line.substr(equals + 1)).second) {
            throw std::runtime_error("Duplicate metadata key: " + line.substr(0, equals));
        }
    }
    return result;
}

void requireFormat(const Metadata& metadata, const std::string& key, const std::string& expected)
{
    const auto entry = metadata.find(key);
    if (entry == metadata.end() || entry->second != expected) {
        throw std::runtime_error("Fixture requires " + key + "=" + expected + "; generate matching fixture metadata");
    }
}

size_t parseSize(const std::string& text, size_t maximum = std::numeric_limits<size_t>::max())
{
    if (text.empty()) {
        throw std::runtime_error("Expected a nonnegative decimal integer");
    }
    size_t value = 0;
    for (char digit : text) {
        if (digit < '0' || digit > '9' || value > maximum / 10 ||
            (value == maximum / 10 && size_t(digit - '0') > maximum % 10)) {
            throw std::runtime_error("Invalid or out-of-range nonnegative decimal integer: " + text);
        }
        value = value * 10 + size_t(digit - '0');
    }
    return value;
}

int32_t parseInteger(const std::string& text) { return static_cast<int32_t>(parseSize(text, INT32_MAX)); }

struct RunnerOptions {
    std::filesystem::path directory;
    int32_t coreLimit = 0;
    bool validateOnly = false;
    bool benchmark = false;
    size_t samples = 1;
    size_t warmup = 0;
    size_t flushBytes = 0;
};

int32_t integer(const Metadata& metadata, const std::string& key, int32_t fallback = -1)
{
    const auto entry = metadata.find(key);
    if (entry == metadata.end()) {
        if (fallback < 0) {
            throw std::runtime_error("Missing metadata key: " + key);
        }
        return fallback;
    }
    return parseInteger(entry->second);
}

float real(const Metadata& metadata, const std::string& key, float fallback, bool nonnegative = true)
{
    const auto entry = metadata.find(key);
    if (entry == metadata.end()) {
        return fallback;
    }
    size_t end = 0;
    const float value = std::stof(entry->second, &end);
    if (end != entry->second.size() || !std::isfinite(value) || (nonnegative && value < 0)) {
        throw std::runtime_error(
            "Expected finite " + std::string(nonnegative ? "nonnegative " : "") + "number for " + key);
    }
    return value;
}

bool boolean(const Metadata& metadata, const std::string& key, bool fallback = false)
{
    const int32_t value = integer(metadata, key, fallback);
    if (value > 1) {
        throw std::runtime_error("Boolean metadata must be 0 or 1: " + key);
    }
    return value != 0;
}

FlashMLABf16Config readConfig(const Metadata& metadata)
{
    FlashMLABf16Config c;
    c.batchSize = integer(metadata, "batch_size");
    c.queryLength = integer(metadata, "query_length");
    c.numHeads = integer(metadata, "num_heads");
    c.headDim = integer(metadata, "head_dim");
    c.pageSize = integer(metadata, "page_size");
    c.numPages = integer(metadata, "num_pages");
    c.softmaxScale = real(metadata, "softmax_scale", 1.0f / std::sqrt(512.0f), false);
    c.topK = integer(metadata, "top_k", 0);
    c.extraTopK = integer(metadata, "extra_top_k", 0);
    c.extraNumPages = integer(metadata, "extra_num_pages", 0);
    c.extraPageSize = integer(metadata, "extra_page_size", c.pageSize);
    if (c.batchSize == 0 || c.queryLength == 0 || (c.numHeads != 64 && c.numHeads != 128) || c.headDim != 512 ||
        c.pageSize == 0 || c.numPages == 0 || (c.extraNumPages != 0 && c.extraPageSize == 0)) {
        throw std::runtime_error("Unsupported shape: require B,Q,pages,page sizes > 0, H=64/128, D=512");
    }
    if ((c.extraTopK != 0) != (c.extraNumPages != 0)) {
        throw std::runtime_error("extra_top_k and extra_num_pages must be both zero or both positive");
    }
    if (checkedSize({size_t(c.batchSize), size_t(c.queryLength), size_t(c.numHeads / 64)}) >
        std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Logical jobs exceed unsigned 32-bit kernel indexing");
    }
    if (checkedSize({size_t(c.numPages), size_t(c.pageSize)}) > INT32_MAX ||
        checkedSize({size_t(c.extraNumPages), size_t(c.extraPageSize)}) > INT32_MAX) {
        throw std::runtime_error("Cache token addresses exceed signed 32-bit kernel indexing");
    }
    return c;
}

template <typename T>
std::vector<T> readTensor(const std::filesystem::path& path, size_t count)
{
    const size_t bytes = checkedSize({count, sizeof(T)});
    if (std::filesystem::file_size(path) != bytes || bytes > size_t(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Tensor size mismatch: " + path.string());
    }
    std::vector<T> result(count);
    std::ifstream input(path, std::ios::binary);
    if (!input || (bytes != 0 && !input.read(reinterpret_cast<char*>(result.data()), bytes))) {
        throw std::runtime_error("Cannot read tensor: " + path.string());
    }
    return result;
}

template <typename T>
void writeTensor(const std::filesystem::path& path, const std::vector<T>& values)
{
    std::ofstream output(path, std::ios::binary);
    const size_t bytes = checkedSize({values.size(), sizeof(T)});
    if (!output || !output.write(reinterpret_cast<const char*>(values.data()), bytes)) {
        throw std::runtime_error("Cannot write tensor: " + path.string());
    }
}

void validateIndices(
    const std::vector<int32_t>& indices, const std::vector<int32_t>& lengths, int32_t topK, int32_t capacity,
    int32_t queryLength)
{
    for (size_t b = 0; b < lengths.size(); ++b) {
        if (lengths[b] < 0 || lengths[b] > topK) {
            throw std::runtime_error("topk_length is outside [0,top_k]");
        }
        for (int32_t q = 0; q < queryLength; ++q) {
            for (int32_t k = 0; k < lengths[b]; ++k) {
                const int32_t index = indices[(b * queryLength + q) * topK + k];
                if (index < -1 || index >= capacity) {
                    throw std::runtime_error("Sparse physical index is outside cache capacity");
                }
            }
        }
    }
}

void validateCache(const std::vector<uint8_t>& cache)
{
    for (size_t record = 0; record < cache.size(); record += CACHE_RECORD_BYTES) {
        for (size_t scale = 256; scale < CACHE_RECORD_BYTES; ++scale) {
            if ((cache[record + scale] & 0x7f) == 0x7f) {
                throw std::runtime_error("Cache contains a reserved E4M3 NaN encoding");
            }
        }
    }
}

float fromBfloat16(uint16_t value)
{
    const uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool compare(
    const std::vector<float>& actual, const std::vector<float>& expected, const std::string& label, double atol,
    double rtol)
{
    double maxAbsolute = 0, maxRelative = 0, squareSum = 0;
    size_t failures = 0, nonfiniteFailures = 0, finiteCount = 0, maxIndex = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e)) {
            // NaN never matches. Infinity requires the same sign.
            if (!(std::isinf(a) && std::isinf(e) && a == e)) {
                ++failures;
                ++nonfiniteFailures;
            }
            continue;
        }
        const double absolute = std::abs(a - e);
        const double relative = absolute / std::max(std::abs(e), 1e-30);
        if (absolute > maxAbsolute) {
            maxAbsolute = absolute;
            maxIndex = i;
        }
        maxRelative = std::max(maxRelative, relative);
        squareSum += absolute * absolute;
        ++finiteCount;
        if (absolute > atol + rtol * std::abs(e)) {
            if (failures < 4) {
                std::cerr << label << " mismatch index=" << i << " actual=" << a << " expected=" << e << '\n';
            }
            ++failures;
        }
    }
    const char* status = failures == 0 ? "PASS" : "FAIL";
    std::cout << label << ": " << status << " mismatches=" << failures << " nonfinite_mismatches=" << nonfiniteFailures
              << " max_abs=" << maxAbsolute << " at=" << maxIndex << " max_rel=" << maxRelative
              << " rmse=" << std::sqrt(squareSum / std::max<size_t>(finiteCount, 1)) << " atol=" << atol
              << " rtol=" << rtol << '\n';
    return failures == 0;
}

class Runtime {
public:
    aclrtStream stream = nullptr;
    Runtime()
    {
        checkAcl(aclInit(nullptr), "aclInit");
        initialized_ = true;
        try {
            checkAcl(aclrtSetDevice(0), "aclrtSetDevice");
            deviceSet_ = true;
            checkAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~Runtime() { cleanup(); }

private:
    bool initialized_ = false;
    bool deviceSet_ = false;
    void cleanup()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet_) {
            aclrtResetDevice(0);
        }
        if (initialized_) {
            aclFinalize();
        }
    }
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes, const char* operation = "aclrtMalloc") : size_(bytes)
    {
        if (bytes != 0) {
            checkAcl(aclrtMalloc(&pointer_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), operation);
        }
    }
    template <typename T>
    explicit DeviceBuffer(const std::vector<T>& values) : DeviceBuffer(checkedSize({values.size(), sizeof(T)}))
    {
        if (size_ != 0) {
            try {
                checkAcl(aclrtMemcpy(pointer_, size_, values.data(), size_, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
            } catch (...) {
                aclrtFree(pointer_);
                throw;
            }
        }
    }
    ~DeviceBuffer()
    {
        if (pointer_ != nullptr) {
            aclrtFree(pointer_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T = void>
    T* get() const
    {
        return static_cast<T*>(pointer_);
    }
    template <typename T>
    std::vector<T> download() const
    {
        std::vector<T> values(size_ / sizeof(T));
        checkAcl(aclrtMemcpy(values.data(), size_, pointer_, size_, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        return values;
    }

private:
    size_t size_;
    void* pointer_ = nullptr;
};

int run(const RunnerOptions& options)
{
    const auto& directory = options.directory;
    const auto m = readMetadata(directory / "fixture.txt");
    requireFormat(m, "query_format", "bf16");
    requireFormat(m, "cache_format", CACHE_FORMAT);
    requireFormat(m, "pv_format", PV_FORMAT);
    const int32_t fixturePScale = integer(m, "probability_scale");
    if (fixturePScale != 1) {
        throw std::runtime_error(
            "Fixture probability_scale=" + std::to_string(fixturePScale) +
            " requires direct BF16 probabilities (probability_scale=1)");
    }
    const int32_t fixtureTileN = integer(m, "tile_n", 128);
    if (fixtureTileN != FLASH_MLA_TILE_N) {
        throw std::runtime_error(
            "Fixture tile_n=" + std::to_string(fixtureTileN) +
            " does not match compiled tile_n=" + std::to_string(FLASH_MLA_TILE_N));
    }
    const auto c = readConfig(m);
    const size_t rows = checkedSize({size_t(c.batchSize), size_t(c.queryLength), size_t(c.numHeads)});
    const size_t elements = checkedSize({rows, size_t(c.headDim)});
    const auto q = readTensor<uint16_t>(directory / "q.bin", elements);
    const auto cache = readTensor<uint8_t>(
        directory / "kv_cache.bin", checkedSize({size_t(c.numPages), size_t(c.pageSize), CACHE_RECORD_BYTES}));
    if (std::any_of(q.begin(), q.end(), [](uint16_t code) { return (code & 0x7f80) == 0x7f80; })) {
        throw std::runtime_error("Query contains a nonfinite BF16 value");
    }
    validateCache(cache);
    std::vector<float> sink;
    if (boolean(m, "has_attn_sink")) {
        sink = readTensor<float>(directory / "attn_sink.bin", c.numHeads);
        for (float value : sink) {
            if (std::isnan(value)) {
                throw std::runtime_error("Attention sink must not contain NaN logits");
            }
        }
    }
    std::vector<int32_t> indices, topkLength, extraIndices, extraTopkLength;
    std::vector<uint8_t> extraCache;
    {
        indices = readTensor<int32_t>(
            directory / "indices.bin", checkedSize({size_t(c.batchSize), size_t(c.queryLength), size_t(c.topK)}));
        std::vector<int32_t> effectiveLengths(c.batchSize, c.topK);
        if (boolean(m, "has_topk_length")) {
            topkLength = readTensor<int32_t>(directory / "topk_length.bin", c.batchSize);
            effectiveLengths = topkLength;
        }
        validateIndices(indices, effectiveLengths, c.topK, c.numPages * c.pageSize, c.queryLength);
        if (c.extraTopK != 0) {
            extraCache = readTensor<uint8_t>(
                directory / "extra_kv_cache.bin",
                checkedSize({size_t(c.extraNumPages), size_t(c.extraPageSize), CACHE_RECORD_BYTES}));
            validateCache(extraCache);
            extraIndices = readTensor<int32_t>(
                directory / "extra_indices.bin",
                checkedSize({size_t(c.batchSize), size_t(c.queryLength), size_t(c.extraTopK)}));
            effectiveLengths.assign(c.batchSize, c.extraTopK);
            if (boolean(m, "has_extra_topk_length")) {
                extraTopkLength = readTensor<int32_t>(directory / "extra_topk_length.bin", c.batchSize);
                effectiveLengths = extraTopkLength;
            }
            validateIndices(
                extraIndices, effectiveLengths, c.extraTopK, c.extraNumPages * c.extraPageSize, c.queryLength);
        }
    }
    const auto mathBits = readTensor<uint16_t>(directory / "golden.bin", elements);
    const auto deviceBits = readTensor<uint16_t>(directory / "device_ref.bin", elements);
    const auto mathLse = readTensor<float>(directory / "golden_lse.bin", rows);
    const auto deviceLse = readTensor<float>(directory / "device_ref_lse.bin", rows);
    if (options.validateOnly) {
        std::cout << "Fixture metadata and tensor sizes validated (tile_n=" << FLASH_MLA_TILE_N
                  << ", probability_scale=1"
                  << ", stamp_compute=" << FLASH_MLA_MARK_STAMP_COMPUTE_PIPE
                  << ", stamp_data=" << FLASH_MLA_MARK_STAMP_DATA_PIPE << ", cache_format=" << CACHE_FORMAT
                  << ", pv_format=" << PV_FORMAT << ")\n";
        return 0;
    }
    Runtime runtime;
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr || platform->GetCoreNumAic() == 0) {
        throw std::runtime_error("Cannot determine platform AIC count");
    }
    const uint32_t jobs = static_cast<uint64_t>(c.batchSize) * c.queryLength * (c.numHeads / 64);
    const uint32_t available = platform->GetCoreNumAic();
    const uint32_t launchCount =
        std::min(jobs, options.coreLimit == 0 ? available : std::min(available, uint32_t(options.coreLimit)));
    std::cout << "FlashMLA BF16 sparse jobs=" << jobs << " launch_count=" << launchCount
              << " platform_aics=" << available << " tile_n=" << FLASH_MLA_TILE_N
              << " gather_slice=" << flash_mla_bf16::K_CHUNK << " gather=paired"
              << " qk_unit_flag=1 pv_unit_flag=1"
              << " query_format=bf16 cache_format=" << CACHE_FORMAT << " pv_format=" << PV_FORMAT << '\n';
    DeviceBuffer dQ(q), dCache(cache), dSink(sink);
    DeviceBuffer dIndices(indices), dTopkLength(topkLength), dExtraCache(extraCache), dExtraIndices(extraIndices),
        dExtraTopkLength(extraTopkLength);
    DeviceBuffer dOutput(checkedSize({elements, sizeof(uint16_t)})), dLse(checkedSize({rows, sizeof(float)}));
    DeviceBuffer flushBuffer(
        options.flushBytes, "aclrtMalloc L2 eviction buffer (reduce --flush-bytes or use --flush-bytes 0)");
    const size_t launches = options.warmup + options.samples;
    if (options.benchmark) {
        std::cout << "BENCH config jobs=" << jobs << " launches=" << launches << " warmup=" << options.warmup
                  << " samples=" << options.samples << " flush_bytes=" << options.flushBytes << '\n';
    }
    for (size_t i = 0; i < launches; ++i) {
        // Poison every launch to detect skipped stores, including empty selections.
        checkAcl(
            aclrtMemset(dOutput.get(), elements * sizeof(uint16_t), 0xff, elements * sizeof(uint16_t)),
            "poison output");
        checkAcl(aclrtMemset(dLse.get(), rows * sizeof(float), 0xff, rows * sizeof(float)), "poison LSE");
        // An oversized memset attempts L2 eviction; hardware cache policy determines its effect.
        if (options.flushBytes != 0) {
            checkAcl(aclrtMemset(flushBuffer.get(), options.flushBytes, 0, options.flushBytes), "L2 eviction memset");
        }
        LaunchFlashMLABf16(
            dOutput.get(), dLse.get<float>(), dQ.get<uint16_t>(), dCache.get<uint8_t>(), dSink.get<float>(),
            dIndices.get<int32_t>(), dTopkLength.get<int32_t>(), dExtraCache.get<uint8_t>(),
            dExtraIndices.get<int32_t>(), dExtraTopkLength.get<int32_t>(), c, launchCount, runtime.stream);
        checkAcl(aclrtSynchronizeStream(runtime.stream), "aclrtSynchronizeStream");
        if (options.benchmark) {
            std::cout << "BENCH launch " << i << " " << (i < options.warmup ? "warmup" : "sample") << " done"
                      << std::endl;
        }
    }
    const auto outputBits = dOutput.download<uint16_t>();
    const auto outputLse = dLse.download<float>();
    // Write device outputs to this invocation's working directory.
    writeTensor(std::filesystem::current_path() / "output.bin", outputBits);
    writeTensor(std::filesystem::current_path() / "output_lse.bin", outputLse);
    std::vector<float> output(elements), math(elements), device(elements);
    std::transform(outputBits.begin(), outputBits.end(), output.begin(), fromBfloat16);
    std::transform(mathBits.begin(), mathBits.end(), math.begin(), fromBfloat16);
    std::transform(deviceBits.begin(), deviceBits.end(), device.begin(), fromBfloat16);
    bool passed = true;
    for (int32_t b = 0; b < c.batchSize; ++b) {
        for (int32_t h = 0; h < c.numHeads; ++h) {
            for (int32_t qIndex = 0; qIndex < c.queryLength; ++qIndex) {
                const size_t lseIndex = (size_t(b) * c.numHeads + h) * c.queryLength + qIndex;
                if (mathLse[lseIndex] == std::numeric_limits<float>::infinity()) {
                    const size_t start = ((size_t(b) * c.queryLength + qIndex) * c.numHeads + h) * c.headDim;
                    for (int32_t d = 0; d < c.headDim; ++d) {
                        if (output[start + d] != 0.0f) {
                            std::cerr << "Empty selection must have exactly zero output at index=" << start + d << '\n';
                            passed = false;
                            break;
                        }
                    }
                }
            }
        }
    }
    passed &= compare(output, math, "mathematical output", real(m, "math_atol", 0.03f), real(m, "math_rtol", 0.08f));
    passed &= compare(
        outputLse, mathLse, "mathematical LSE", real(m, "math_lse_atol", 0.03f), real(m, "math_lse_rtol", 0.01f));
    passed &=
        compare(output, device, "device-model output", real(m, "device_atol", 0.01f), real(m, "device_rtol", 0.03f));
    passed &= compare(
        outputLse, deviceLse, "device-model LSE", real(m, "device_lse_atol", 0.01f),
        real(m, "device_lse_rtol", 0.001f));
    std::cout << "Implementation correctness: " << (passed ? "PASS" : "FAIL")
              << " (mathematical and device references, nonfinite agreement, exact empty output)\n";
    return passed ? 0 : 1;
}
} // namespace

int runFlashMla(int argc, char** argv, bool benchmark)
{
    try {
        RunnerOptions options;
        options.benchmark = benchmark;
        if (benchmark) {
            options.samples = 5;
            options.warmup = 2;
            options.flushBytes = 8000000000ULL;
        }
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") {
                std::cout << "Usage: " << (benchmark ? "flash_mla_bf16_bench" : "flash_mla_bf16_test")
                          << " --golden-dir DIR [--core-limit N] [--validate-only]";
                if (benchmark) {
                    std::cout << " [--samples N] [--warmup N] [--flush-bytes N]\n"
                                 "Defaults: --samples 5 --warmup 2 --flush-bytes 8000000000\n"
                                 "Use --warmup 0 to skip warmup or --flush-bytes 0 to disable eviction attempts.";
                }
                std::cout << '\n';
                return 0;
            }
            if (argument == "--validate-only") {
                options.validateOnly = true;
            } else if (
                (argument == "--golden-dir" || argument == "--core-limit" ||
                 (benchmark && (argument == "--samples" || argument == "--warmup" || argument == "--flush-bytes"))) &&
                i + 1 < argc) {
                const std::string value = argv[++i];
                if (argument == "--golden-dir") {
                    options.directory = value;
                } else if (argument == "--core-limit") {
                    options.coreLimit = parseInteger(value);
                } else if (argument == "--samples") {
                    options.samples = parseSize(value, std::numeric_limits<uint32_t>::max());
                } else if (argument == "--warmup") {
                    options.warmup = parseSize(value, std::numeric_limits<uint32_t>::max());
                } else {
                    options.flushBytes = parseSize(value);
                }
            } else {
                throw std::runtime_error("Unknown or incomplete argument: " + argument);
            }
        }
        if (options.directory.empty()) {
            throw std::runtime_error("--golden-dir is required");
        }
        if (options.samples == 0) {
            throw std::runtime_error("--samples must be positive");
        }
        if (options.samples > std::numeric_limits<uint32_t>::max() - options.warmup) {
            throw std::runtime_error("--samples plus --warmup exceeds the unsigned 32-bit launch limit");
        }
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << "FlashMLA BF16: " << error.what() << '\n';
        return 2;
    }
}
