/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <string>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <sys/stat.h>
#ifndef __CPU_SIM
#ifndef __COSTMODEL
#include "acl/acl.h"
#endif
#endif
#include <pto/common/type.hpp>

namespace PtoTestCommon {

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

#define PAD_VALUE_NULL (-100)
#define PAD_VALUE_MAX (1)
#define PAD_VALUE_MIN (-1)

#define CHECK_RESULT_GTEST(x) \
    if (!(x))                 \
        ASSERT_TRUE(false);

typedef enum
{
    DT_UNDEFINED = -1,
    FLOAT = 0,
    HALF = 1,
    INT8_T = 2,
    INT32_T = 3,
    UINT8_T = 4,
    INT16_T = 6,
    UINT16_T = 7,
    UINT32_T = 8,
    INT64_T = 9,
    UINT64_T = 10,
    DOUBLE = 11,
    BOOL = 12,
    STRING = 13,
    COMPLEX64 = 16,
    COMPLEX128 = 17,
    BF16 = 27
} printDataType;

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("Failed to get file. Path = %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. Path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("%s: file size (%lu) is larger than buffer size (%lu)", filePath.c_str(), size, bufferSize);
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

#ifdef _WIN32
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_BINARY, S_IRUSR | S_IWRITE);
#else
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
#endif
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void)close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

template <typename T>
void DoPrintData(const T *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(5) << data[i];
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void DoPrintHalfData(const aclFloat16 *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(5) << std::setprecision(6) <<
#if defined(__CPU_SIM) || defined(__COSTMODEL)
            (float)data[i];
#else
            aclFloat16ToFloat(data[i]);
#endif
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void PrintData(const void *data, size_t count, printDataType dataType, size_t elementsPerRow = 16)
{
    if (data == nullptr) {
        ERROR_LOG("Print data failed. data is nullptr");
        return;
    }

    switch (dataType) {
        case BOOL:
            DoPrintData(reinterpret_cast<const bool *>(data), count, elementsPerRow);
            break;
        case INT8_T:
            DoPrintData(reinterpret_cast<const int8_t *>(data), count, elementsPerRow);
            break;
        case UINT8_T:
            DoPrintData(reinterpret_cast<const uint8_t *>(data), count, elementsPerRow);
            break;
        case INT16_T:
            DoPrintData(reinterpret_cast<const int16_t *>(data), count, elementsPerRow);
            break;
        case UINT16_T:
            DoPrintData(reinterpret_cast<const uint16_t *>(data), count, elementsPerRow);
            break;
        case INT32_T:
            DoPrintData(reinterpret_cast<const int32_t *>(data), count, elementsPerRow);
            break;
        case UINT32_T:
            DoPrintData(reinterpret_cast<const uint32_t *>(data), count, elementsPerRow);
            break;
        case INT64_T:
            DoPrintData(reinterpret_cast<const int64_t *>(data), count, elementsPerRow);
            break;
        case UINT64_T:
            DoPrintData(reinterpret_cast<const uint64_t *>(data), count, elementsPerRow);
            break;

        case HALF:
            DoPrintHalfData(reinterpret_cast<const aclFloat16 *>(data), count, elementsPerRow);
            break;

        case FLOAT:
            DoPrintData(reinterpret_cast<const float *>(data), count, elementsPerRow);
            break;
        case DOUBLE:
            DoPrintData(reinterpret_cast<const double *>(data), count, elementsPerRow);
            break;
        default:
            ERROR_LOG("Unsupported type: %d", dataType);
    }
    std::cout << std::endl;
}

#define RESET "\033[0m"
#define BOLD_RED "\033[1;31m"

template <typename T>
bool ResultCmp(const std::vector<T> &outDataValExp, const T *outDataValAct, float eps, size_t threshold = 0,
               size_t zeroCountThreshold = 1000, bool printAll = false, bool printErr = false, size_t testNum = 0)
{
    threshold = threshold == 0 ? static_cast<int>(outDataValExp.size() * eps) : threshold;

    float maxDiff = 0;
    float maxDiffRatio = 0;
    size_t zeroCount = 0;
    size_t errCount = 0;

    bool rst = true;
    size_t eSize = outDataValExp.size();
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);
        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        maxDiffRatio = std::max(relRatio, maxDiffRatio);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;
        testNum = testNum - (testNum > 0 ? 1 : 0);

        auto eErr = ((diff > eps && relRatio > eps) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if ((printAll) || (eErr && printErr) || (testNum > 0)) {
            std::cout << (eErr ? BOLD_RED : "") << "idx: 0x" << eIdx << ", exp->" << expVal << ", act->" << actVal
                      << ", diff->" << diff << ", diff ratio->" << relRatio << ", zero count->0x" << zeroCount
                      << (eErr ? (" [ERROR]" RESET) : "") << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
    }

    float errCountRatio = static_cast<float>(errCount) / static_cast<float>(eSize);
    float zeroCountRatio = static_cast<float>(zeroCount) / static_cast<float>(eSize);
    std::cout << "max diff: " << maxDiff << ", diff threshold: " << eps << ", max diff ratio: " << maxDiffRatio
              << ", err count: " << errCount << ", err threshold: " << threshold
              << ", err count ratio: " << errCountRatio << ", act zero count: 0x" << zeroCount
              << ", act zero threshold: 0x" << zeroCountThreshold << ", act zero ratio: " << zeroCountRatio
              << std::endl;
    if (rst || printAll || printErr) {
        return rst;
    }

    errCount = 0;
    zeroCount = 0;
    for (size_t eIdx = 0; eIdx < eSize; eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);

        auto diff = std::abs(expVal - actVal);
        auto relRatio = std::abs(diff / expVal);
        zeroCount += std::abs(actVal - 0.0f) <= 1e-6 and std::abs(expVal - 0.0f) > 1e-6 ? 1 : 0;

        auto eErr = ((diff > eps && relRatio > eps) || (zeroCount > zeroCountThreshold));
        errCount += eErr ? 1 : 0;

        if (std::isnan(expVal) || std::isnan(actVal)) {
            std::cout << "idx: " << eIdx << ", exp->" << expVal << ", act->" << actVal << std::endl;
        }

        if (eErr) {
            std::cout << BOLD_RED << "idx: 0x" << eIdx << ", exp->" << expVal << ", act->" << actVal << ", diff->"
                      << diff << ", diff ratio->" << relRatio << ", zero count->0x" << zeroCount << " [ERROR]" RESET
                      << std::endl;
        }
        rst = !((errCount > threshold || zeroCount > zeroCountThreshold));
        if (!rst) {
            break;
        }
    }
    return false;
}
template <typename T>
bool ResultCmp(const std::vector<T> &outDataValExp, const std::vector<T> &outDataValAct, float eps,
               size_t threshold = 0, size_t zeroCountThreshold = 1000, bool printAll = false, bool printErr = false,
               size_t testNum = 0)
{
    if (outDataValExp.size() != outDataValAct.size()) {
        std::cout << "out size is not eq, golden: " << outDataValExp.size() << ", act: " << outDataValAct.size()
                  << std::endl;
        return false;
    }
    return ResultCmp(outDataValExp, outDataValAct.data(), eps, threshold, zeroCountThreshold, printAll, printErr,
                     testNum);
}

} // namespace PtoTestCommon
