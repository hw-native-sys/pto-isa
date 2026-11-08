/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_LINX_COMPAT_HPP
#define PTO_NPU_LINX_COMPAT_HPP

// The Linx compiler owns the tile ABI and does not use CANN's address-space
// spellings. Keep the shared PTO declarations source-compatible while Linx
// headers provide the actual Tile/GlobalTensor representation.
#ifndef AICORE
#define AICORE
#endif

#ifndef __gm__
#define __gm__
#endif
#ifndef __ubuf__
#define __ubuf__
#endif
#ifndef __cbuf__
#define __cbuf__
#endif
#ifndef __ca__
#define __ca__
#endif
#ifndef __cb__
#define __cb__
#endif
#ifndef __cc__
#define __cc__
#endif
#ifndef __fbuf__
#define __fbuf__
#endif
#ifndef __biasbuf__
#define __biasbuf__
#endif

#ifndef PTO_INLINE
#define PTO_INLINE inline __attribute__((always_inline))
#endif
#ifndef PTO_INST
#define PTO_INST PTO_INLINE
#endif
#ifndef PTO_INTERNAL
#define PTO_INTERNAL PTO_INLINE
#endif
#ifndef OP_NAME
#define OP_NAME(Name)
#endif
#ifndef OP_TYPE
#define OP_TYPE(TypeName)
#endif

#ifndef PTO_STATIC_ASSERT
#define PTO_STATIC_ASSERT(...) static_assert(__VA_ARGS__)
#endif
#ifndef PTO_CPU_ASSERT
#define PTO_CPU_ASSERT(...) ((void)0)
#endif

namespace pto {
// Linx block ordering provides the target-side synchronization. Keep the
// public event return type available for shared call sites; Linx operations
// do not create CANN pipe events.
struct RecordEvent {};
} // namespace pto

#define PTO_LINX_TARGET 1

#endif // PTO_NPU_LINX_COMPAT_HPP
