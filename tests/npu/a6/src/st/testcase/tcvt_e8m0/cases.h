/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCVT_E8M0_CASES_H_
#define TCVT_E8M0_CASES_H_

// name, source rows/columns, destination rows/columns, valid rows/columns,
// direction (true: bf16 -> e8m0, false: e8m0 -> bf16), dynamic valid extents.
#define E8M0_CVT_CASES(M)                                      \
    M(Bf16ToE8m0Full, 4, 512, 4, 512, 4, 512, true, false)     \
    M(Bf16ToE8m0Tail, 1, 512, 1, 512, 1, 257, true, true)      \
    M(Bf16ToE8m0Strided, 5, 640, 7, 704, 3, 579, true, false)  \
    M(Bf16ToE8m0Dynamic, 5, 640, 7, 704, 3, 579, true, true)   \
    M(Bf16ToE8m0Wide, 3, 1088, 5, 1152, 2, 1025, true, false)  \
    M(E8m0ToBf16Full, 4, 512, 4, 512, 4, 512, false, false)    \
    M(E8m0ToBf16Tail, 1, 512, 1, 512, 1, 257, false, true)     \
    M(E8m0ToBf16Strided, 5, 640, 7, 704, 3, 579, false, false) \
    M(E8m0ToBf16Dynamic, 5, 640, 7, 704, 3, 579, false, true)  \
    M(E8m0ToBf16Wide, 3, 1088, 5, 1152, 2, 1025, false, false)

struct E8m0CvtCase {
    const char* name;
    int sourceRows;
    int sourceCols;
    int destinationRows;
    int destinationCols;
    int validRows;
    int validCols;
    bool toE8m0;
    bool dynamic;
};

#define E8M0_CVT_DESCRIPTION(Name, SR, SC, DR, DC, VR, VC, ToE8m0, Dynamic) \
    {#Name, SR, SC, DR, DC, VR, VC, ToE8m0, Dynamic},
constexpr E8m0CvtCase kE8m0CvtCases[] = {E8M0_CVT_CASES(E8M0_CVT_DESCRIPTION)};
#undef E8M0_CVT_DESCRIPTION

enum class E8m0CvtCaseId {
#define E8M0_CVT_ENUM(Name, ...) Name,
    E8M0_CVT_CASES(E8M0_CVT_ENUM)
#undef E8M0_CVT_ENUM
};

void launchE8m0Cvt(
    int caseId, unsigned char* dst, unsigned char* src, int rows, int cols, int round, int api, void* stream);
void launchE8m0ToBf16AllBytes(unsigned char* dst, unsigned char* src, void* stream);
void launchBf16ToE8m0Specials(unsigned char* dst, unsigned char* src, int round, void* stream);

#endif // TCVT_E8M0_CASES_H_
