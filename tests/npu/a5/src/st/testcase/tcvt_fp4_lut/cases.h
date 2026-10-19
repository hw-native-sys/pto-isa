/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCVT_FP4_LUT_CASES_H_
#define TCVT_FP4_LUT_CASES_H_

// name, source rows/columns, destination rows/columns, valid rows/columns,
// NZ destination, compact Normal, dynamic valid extents.
#define FP4_LUT_CASES(M)                                                \
    M(MainND, 32, 512, 32, 512, 32, 512, false, false, false)           \
    M(MainNZ, 32, 512, 32, 512, 32, 512, true, false, false)            \
    M(MinimumND, 1, 64, 1, 64, 1, 64, false, false, false)              \
    M(MinimumNZ, 1, 64, 16, 64, 1, 64, true, false, false)              \
    M(SingleNibbleND, 1, 64, 1, 96, 1, 1, false, false, true)           \
    M(SingleNibbleNZ, 1, 64, 32, 96, 1, 1, true, true, true)            \
    M(StridedStaticND, 5, 640, 7, 704, 3, 579, false, false, false)     \
    M(StridedDynamicND, 5, 640, 7, 704, 3, 579, false, false, true)     \
    M(StridedStaticNZ, 5, 640, 32, 704, 3, 579, true, false, false)     \
    M(StridedDynamicNZ, 5, 640, 32, 704, 3, 579, true, false, true)     \
    M(CompactStaticNZ, 5, 640, 32, 704, 3, 579, true, true, false)      \
    M(CompactDynamicNZ, 5, 640, 32, 704, 3, 579, true, true, true)      \
    M(RectangularNZ, 19, 128, 48, 192, 17, 93, true, false, false)      \
    M(RectangularCompactNZ, 19, 128, 48, 192, 17, 93, true, true, true) \
    M(WideND, 3, 1088, 5, 1152, 2, 1025, false, false, false)           \
    M(WideNZ, 3, 1088, 32, 1152, 2, 1025, true, false, false)           \
    M(SecondRegisterTailND, 2, 512, 4, 544, 1, 257, false, false, true) \
    M(SecondRegisterTailNZ, 2, 512, 32, 544, 1, 257, true, false, true)

struct Fp4LutCase {
    const char* name;
    int sourceRows;
    int sourceCols;
    int destinationRows;
    int destinationCols;
    int validRows;
    int validCols;
    bool nz;
    bool compact;
    bool dynamic;
};

#define FP4_LUT_DESCRIPTION(Name, SR, SC, DR, DC, VR, VC, NZ, Compact, Dynamic) \
    {#Name, SR, SC, DR, DC, VR, VC, NZ, Compact, Dynamic},
constexpr Fp4LutCase kFp4LutCases[] = {FP4_LUT_CASES(FP4_LUT_DESCRIPTION)};
#undef FP4_LUT_DESCRIPTION

enum class Fp4LutCaseId {
#define FP4_LUT_ENUM(Name, ...) Name,
    FP4_LUT_CASES(FP4_LUT_ENUM)
#undef FP4_LUT_ENUM
};

void launchFp4Lut(
    int caseId, unsigned char* dst, unsigned char* src, int rows, int cols, int round, int api, void* stream);
void launchFp4LutRegression(int pair, unsigned char* dst, unsigned char* src, void* stream);

#endif
