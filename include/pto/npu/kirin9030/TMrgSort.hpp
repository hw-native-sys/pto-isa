/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TMRGSORT_KIRIN9030_HPP
#define TMRGSORT_KIRIN9030_HPP
#define MOV_UB_2_UB_STUB
template <typename DstTileData>
PTO_INTERNAL void MovUb2Ub(__ubuf__ typename DstTileData::DType *dstPtr, __ubuf__ typename DstTileData::DType *tmpPtr,
                           unsigned dstCol)
{
    unsigned lenBurst = (dstCol * sizeof(typename DstTileData::DType) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    copy_ubuf_to_ubuf((__ubuf__ void *)dstPtr, (__ubuf__ void *)tmpPtr, 1, lenBurst, 0, 0);
}
#include "pto/npu/a5/TMrgSort.hpp"
#undef MOV_UB_2_UB_STUB
#endif // TMRGSORT_KIRIN9030_HPP
