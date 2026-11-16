/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Integrated from LinxISA/Linx-TileOP-API; see ../../README.md for provenance.

#ifndef PTO_LINX_VENDOR_PTO_TILEOP_API_REVISION_HPP
#define PTO_LINX_VENDOR_PTO_TILEOP_API_REVISION_HPP

#define PTO_TILEOP_API_VERSION "0.58.7"
#define PTO_TILEOP_API_SPEC_VERSION "0.58.7"

// Pinned upstream snapshot; integration changes are recorded in the backend README.
#define PTO_TILEOP_API_REVISION "ad5a2310664041be1f2fad5be1dd8e9d682f23df"
#define PTO_TILEOP_API_REVISION_IS_EXACT 1

#define PTO_TILEOP_API_HAS_INDEXED_TLSU_RANGE 1

// Feature gates let consumers fail fast without comparing revision strings.
#define PTO_TILEOP_API_HAS_LOCAL_M32_TLOAD_ASSEMBLY 1
#define PTO_TILEOP_API_HAS_SHARED_ND_TO_CUBE 1
#define PTO_TILEOP_API_HAS_LOCAL_B_KN_FIX 1
#define PTO_TILEOP_API_HAS_TEXPDIF 1
#define PTO_TILEOP_API_HAS_ELEMENT_TILE 1
#define PTO_TILEOP_API_HAS_ELEMENT_VIEW_METADATA 1
#define PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS 1
#define PTO_TILEOP_API_HAS_512_INTEGER_ELEMENT_TILE 1
#define PTO_TILEOP_API_HAS_I64_ELEMENT_TILE_LEAF32 1
#define PTO_TILEOP_API_HAS_INTEGER_ELEMENT_STORAGE_1024 1
#define PTO_TILEOP_API_HAS_I32_ELEMENT_STORAGE_2048 1
#define PTO_TILEOP_API_HAS_DENSE_F16_U16_ELEMENT_TILE 1
#define PTO_TILEOP_API_HAS_HIERARCHICAL_ELEMENT_STORAGE 1
#define PTO_TILEOP_API_HAS_NORM_U32_ELEMENT_VIEW_V2 1

#endif
