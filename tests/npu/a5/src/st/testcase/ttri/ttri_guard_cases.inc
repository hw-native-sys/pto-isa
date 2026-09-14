/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// The kind selects native (0), fp16 (1), or bf16 (2) storage interpretation.
PTO_TTRI_GUARD_CASE(guard_int64_s4x4_v4x1_upper, int64_t, 0, 4, 4, 4, 1, 1, 0)
PTO_TTRI_GUARD_CASE(guard_uint64_s4x4_v4x1_lower, uint64_t, 0, 4, 4, 4, 1, 0, 0)
PTO_TTRI_GUARD_CASE(guard_int64_s8x68_v4x65_upper_diag_2, int64_t, 0, 8, 68, 4, 65, 1, 2)
PTO_TTRI_GUARD_CASE(guard_uint64_s8x68_v4x65_lower_diag_n2, uint64_t, 0, 8, 68, 4, 65, 0, -2)
PTO_TTRI_GUARD_CASE(guard_float_s64x8_v64x1_upper, float, 0, 64, 8, 64, 1, 1, 0)
PTO_TTRI_GUARD_CASE(guard_float_s64x8_v64x1_lower, float, 0, 64, 8, 64, 1, 0, 0)
PTO_TTRI_GUARD_CASE(guard_fp16_s160x16_v160x1_upper, uint16_t, 1, 160, 16, 160, 1, 1, 0)
PTO_TTRI_GUARD_CASE(guard_fp16_s160x16_v160x1_lower, uint16_t, 1, 160, 16, 160, 1, 0, 0)
PTO_TTRI_GUARD_CASE(guard_bf16_s160x16_v160x1_upper, uint16_t, 2, 160, 16, 160, 1, 1, 0)
PTO_TTRI_GUARD_CASE(guard_bf16_s160x16_v160x1_lower, uint16_t, 2, 160, 16, 160, 1, 0, 0)
PTO_TTRI_GUARD_CASE(guard_int8_s288x32_v288x1_lower, int8_t, 0, 288, 32, 288, 1, 0, 0)
PTO_TTRI_GUARD_CASE(guard_uint8_s288x32_v288x1_upper, uint8_t, 0, 288, 32, 288, 1, 1, 0)
PTO_TTRI_GUARD_CASE(guard_int16_s160x16_v160x1_lower, int16_t, 0, 160, 16, 160, 1, 0, 0)
PTO_TTRI_GUARD_CASE(guard_uint16_s160x16_v160x1_upper, uint16_t, 0, 160, 16, 160, 1, 1, 0)
PTO_TTRI_GUARD_CASE(guard_int32_s96x8_v96x1_lower, int32_t, 0, 96, 8, 96, 1, 0, 0)
PTO_TTRI_GUARD_CASE(guard_uint32_s96x8_v96x1_upper, uint32_t, 0, 96, 8, 96, 1, 1, 0)
PTO_TTRI_GUARD_CASE(guard_float_s96x72_v96x65_upper_diag_3, float, 0, 96, 72, 96, 65, 1, 3)
PTO_TTRI_GUARD_CASE(guard_float_s96x72_v96x65_lower_diag_n3, float, 0, 96, 72, 96, 65, 0, -3)
PTO_TTRI_GUARD_CASE(guard_int64_diagonal_max, int64_t, 0, 4, 4, 4, 1, 1, INT_MAX)
PTO_TTRI_GUARD_CASE(guard_int64_diagonal_min, int64_t, 0, 4, 4, 4, 1, 1, INT_MIN)
PTO_TTRI_GUARD_CASE(guard_float_diagonal_max, float, 0, 64, 8, 64, 1, 0, INT_MAX)
PTO_TTRI_GUARD_CASE(guard_float_diagonal_min, float, 0, 64, 8, 64, 1, 0, INT_MIN)
