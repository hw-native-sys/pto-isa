/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCVT_FP4_LUT_V2C_CASES_H_
#define TCVT_FP4_LUT_V2C_CASES_H_

#define FP4_LUT_V2C_CASES(X)           \
    X(KNMain, 16, 32, 512, 32, false)  \
    X(KNWrap, 16, 128, 128, 32, false) \
    X(NKWrap, 16, 256, 96, 64, true)

enum class Fp4LutV2cCaseId {
#define FP4_LUT_V2C_ID(Name, ...) Name,
    FP4_LUT_V2C_CASES(FP4_LUT_V2C_ID)
#undef FP4_LUT_V2C_ID
};

struct Fp4LutV2cCase {
    const char* name;
    int m;
    int totalK;
    int n;
    int tileK;
    bool nk;
};

inline constexpr Fp4LutV2cCase kFp4LutV2cCases[] = {
#define FP4_LUT_V2C_CASE(Name, M, TotalK, N, TileK, NK) {#Name, M, TotalK, N, TileK, NK},
    FP4_LUT_V2C_CASES(FP4_LUT_V2C_CASE)
#undef FP4_LUT_V2C_CASE
};

void launchFp4LutV2c(int caseId, unsigned char* output, unsigned char* a, unsigned char* packedB, void* stream);

#endif // TCVT_FP4_LUT_V2C_CASES_H_
