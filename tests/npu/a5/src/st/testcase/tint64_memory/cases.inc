/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

PTO_BOUND_CASE(add_int64_t_end4, int64_t, 0, 4, 4)
#ifndef __CPU_SIM
PTO_BOUND_CASE(adds_int64_t_end4, int64_t, 1, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(not_int64_t_end4, int64_t, 2, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(div_int64_t_end4, int64_t, 3, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(divs_int64_t_end4, int64_t, 4, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rem_int64_t_end4, int64_t, 5, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmax_int64_t_end4, int64_t, 6, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmin_int64_t_end4, int64_t, 7, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowsum_int64_t_end4, int64_t, 8, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(colmax_int64_t_end4, int64_t, 9, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(colmin_int64_t_end4, int64_t, 10, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(colsum_int64_t_end4, int64_t, 11, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(partadd_int64_t_end4, int64_t, 12, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowexpandadd_int64_t_end4, int64_t, 13, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(colexpandadd_int64_t_end4, int64_t, 14, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(sel_int64_t_end4, int64_t, 15, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(sels_int64_t_end4, int64_t, 16, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmp_int64_t_end4, int64_t, 17, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmps_int64_t_end4, int64_t, 18, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(scatter_int64_t_end4, int64_t, 19, 4, 4)
#endif
PTO_BOUND_CASE(add_int64_t_end64_valid64, int64_t, 0, 64, 64)
PTO_BOUND_CASE(add_int64_t_end68_valid65, int64_t, 0, 68, 65)
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmax_int64_t_end64_valid64, int64_t, 6, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmax_int64_t_end68_valid65, int64_t, 6, 68, 65)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmin_int64_t_end64_valid64, int64_t, 7, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmin_int64_t_end68_valid65, int64_t, 7, 68, 65)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowsum_int64_t_end64_valid64, int64_t, 8, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowsum_int64_t_end68_valid65, int64_t, 8, 68, 65)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmp_int64_t_end64_valid64, int64_t, 17, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmp_int64_t_end68_valid65, int64_t, 17, 68, 65)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmps_int64_t_end64_valid64, int64_t, 18, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmps_int64_t_end68_valid65, int64_t, 18, 68, 65)
#endif
PTO_LAYOUT_CASE(gather_index_dn_int64_t_8x8_v8x8, int64_t, 0, 8, 8, 8, 8)
PTO_LAYOUT_CASE(gather_index_dn_int64_t_8x72_v5x65, int64_t, 0, 8, 72, 5, 65)
PTO_LAYOUT_CASE(gather_dst_dn_int64_t_8x8_v8x8, int64_t, 1, 8, 8, 8, 8)
PTO_LAYOUT_CASE(gather_dst_dn_int64_t_8x72_v5x65, int64_t, 1, 8, 72, 5, 65)
PTO_LAYOUT_CASE(scatter_index_dn_int64_t_8x8_v8x8, int64_t, 2, 8, 8, 8, 8)
PTO_LAYOUT_CASE(scatter_index_dn_int64_t_8x72_v5x65, int64_t, 2, 8, 72, 5, 65)
PTO_LAYOUT_CASE(scatter_src_dn_int64_t_8x8_v8x8, int64_t, 3, 8, 8, 8, 8)
PTO_LAYOUT_CASE(scatter_src_dn_int64_t_8x72_v5x65, int64_t, 3, 8, 72, 5, 65)
PTO_BOUND_CASE(add_uint64_t_end4, uint64_t, 0, 4, 4)
#ifndef __CPU_SIM
PTO_BOUND_CASE(adds_uint64_t_end4, uint64_t, 1, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(not_uint64_t_end4, uint64_t, 2, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(div_uint64_t_end4, uint64_t, 3, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(divs_uint64_t_end4, uint64_t, 4, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rem_uint64_t_end4, uint64_t, 5, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmax_uint64_t_end4, uint64_t, 6, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmin_uint64_t_end4, uint64_t, 7, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowsum_uint64_t_end4, uint64_t, 8, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(colmax_uint64_t_end4, uint64_t, 9, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(colmin_uint64_t_end4, uint64_t, 10, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(colsum_uint64_t_end4, uint64_t, 11, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(partadd_uint64_t_end4, uint64_t, 12, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowexpandadd_uint64_t_end4, uint64_t, 13, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(colexpandadd_uint64_t_end4, uint64_t, 14, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(sel_uint64_t_end4, uint64_t, 15, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(sels_uint64_t_end4, uint64_t, 16, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmp_uint64_t_end4, uint64_t, 17, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmps_uint64_t_end4, uint64_t, 18, 4, 4)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(scatter_uint64_t_end4, uint64_t, 19, 4, 4)
#endif
PTO_BOUND_CASE(add_uint64_t_end64_valid64, uint64_t, 0, 64, 64)
PTO_BOUND_CASE(add_uint64_t_end68_valid65, uint64_t, 0, 68, 65)
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmax_uint64_t_end64_valid64, uint64_t, 6, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmax_uint64_t_end68_valid65, uint64_t, 6, 68, 65)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmin_uint64_t_end64_valid64, uint64_t, 7, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowmin_uint64_t_end68_valid65, uint64_t, 7, 68, 65)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowsum_uint64_t_end64_valid64, uint64_t, 8, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(rowsum_uint64_t_end68_valid65, uint64_t, 8, 68, 65)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmp_uint64_t_end64_valid64, uint64_t, 17, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmp_uint64_t_end68_valid65, uint64_t, 17, 68, 65)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmps_uint64_t_end64_valid64, uint64_t, 18, 64, 64)
#endif
#ifndef __CPU_SIM
PTO_BOUND_CASE(cmps_uint64_t_end68_valid65, uint64_t, 18, 68, 65)
#endif
PTO_LAYOUT_CASE(gather_index_dn_uint64_t_8x8_v8x8, uint64_t, 0, 8, 8, 8, 8)
PTO_LAYOUT_CASE(gather_index_dn_uint64_t_8x72_v5x65, uint64_t, 0, 8, 72, 5, 65)
PTO_LAYOUT_CASE(gather_dst_dn_uint64_t_8x8_v8x8, uint64_t, 1, 8, 8, 8, 8)
PTO_LAYOUT_CASE(gather_dst_dn_uint64_t_8x72_v5x65, uint64_t, 1, 8, 72, 5, 65)
PTO_LAYOUT_CASE(scatter_index_dn_uint64_t_8x8_v8x8, uint64_t, 2, 8, 8, 8, 8)
PTO_LAYOUT_CASE(scatter_index_dn_uint64_t_8x72_v5x65, uint64_t, 2, 8, 72, 5, 65)
PTO_LAYOUT_CASE(scatter_src_dn_uint64_t_8x8_v8x8, uint64_t, 3, 8, 8, 8, 8)
PTO_LAYOUT_CASE(scatter_src_dn_uint64_t_8x72_v5x65, uint64_t, 3, 8, 72, 5, 65)
PTO_EMPTY_CASE(extract_offset0_v4x0, 0, 4, 0)
PTO_EMPTY_CASE(extract_offset1_v4x0, 1, 4, 0)
PTO_EMPTY_CASE(extract_offset0_v0x8, 0, 0, 8)
PTO_EMPTY_CASE(extract_offset1_v0x8, 1, 0, 8)

#ifndef __CPU_SIM
PTO_BOUND_CASE(index_end_20_int64_t, int64_t, 20, 4, 4)
PTO_BOUND_CASE(index_end_21_int64_t, int64_t, 21, 4, 4)
PTO_BOUND_CASE(index_end_20_uint64_t, uint64_t, 20, 4, 4)
PTO_BOUND_CASE(index_end_21_uint64_t, uint64_t, 21, 4, 4)
#endif

#ifndef __CPU_SIM
PTO_BOUND_CASE(colexpand_int64_t_end4_valid4, int64_t, 22, 4, 4)
PTO_BOUND_CASE(colexpand_int64_t_end32_valid32, int64_t, 22, 32, 32)
PTO_BOUND_CASE(colexpand_int64_t_end36_valid33, int64_t, 22, 36, 33)
PTO_BOUND_CASE(colexpand_uint64_t_end4_valid4, uint64_t, 22, 4, 4)
PTO_BOUND_CASE(colexpand_uint64_t_end32_valid32, uint64_t, 22, 32, 32)
PTO_BOUND_CASE(colexpand_uint64_t_end36_valid33, uint64_t, 22, 36, 33)
#endif
