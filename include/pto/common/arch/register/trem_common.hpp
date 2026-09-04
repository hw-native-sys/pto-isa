/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file trem_common.hpp
 * @brief Common TREM float/half remainder helpers for Kirin9030, KirinDev0000
 *
 * RemOpBase holds the arch-independent float/half remainder kernels and the
 * bit masks they use. The integer remainder kernel is arch-specific
 * (kirin9030 uses hardware integer vdiv; kirinDev0000 lacks it and falls back
 * to a software division), so each arch shell derives its RemOp from this base
 * and provides RemInt and the BinInstr dispatch itself.
 *
 * The arch shell must include its own arch common.hpp (for RegTensor, MaskReg,
 * vector intrinsics) before this file.
 */

#ifndef TREM_COMMON_REGISTER_HPP
#define TREM_COMMON_REGISTER_HPP

namespace pto {

template <typename T>
struct RemOpBase {
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    static const U inf = sizeof(T) == sizeof(int32_t) ? 0x7F800000 : 0x7C00;
    static const U abs = sizeof(T) == sizeof(int32_t) ? 0x7FFFFFFF : 0x7FFF;
    static const U nan = sizeof(T) == sizeof(int32_t) ? 0x7FC00000 : 0x7E00;
    PTO_INTERNAL static void RemFloat(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg& preg)
    {
        MaskReg infMask, diffSignMask;
        RegTensor<T> diffSign, src0Abs, nanReg, absReg;
        vdiv(dst, src0, src1, preg, MODE_ZEROING);
        vtrc(dst, dst, ROUND_F, preg);
        vmul(dst, dst, src1, preg, MODE_ZEROING);
        vsub(dst, src0, dst, preg, MODE_ZEROING);

        vmul(diffSign, src1, dst, preg, MODE_ZEROING);
        vcmps_lt(diffSignMask, diffSign, 0.0f, preg);
        vadd(diffSign, dst, src1, diffSignMask, MODE_MERGING);

        vdup((RegTensor<U>&)absReg, abs, preg, MODE_ZEROING);
        vand((RegTensor<U>&)src0Abs, (RegTensor<U>&)src0, (RegTensor<U>&)absReg, preg);
        vcmps_eq(infMask, (RegTensor<U>&)src0Abs, inf, preg);
        vdup((RegTensor<U>&)nanReg, nan, infMask, MODE_ZEROING);
        vsel(dst, nanReg, dst, infMask);
    }

    PTO_INTERNAL static void RemHalf(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg& preg)
    {
        MaskReg infMask, diffSignMask;
        RegTensor<float> even0, even1, evenQuotient, odd0, odd1, oddQuotient;
        RegTensor<T> evenDst, oddDst, diffSign, src0Abs, nanReg, absReg;
        vcvt(even0, src0, preg, PART_EVEN);
        vcvt(even1, src1, preg, PART_EVEN);
        vcvt(odd0, src0, preg, PART_ODD);
        vcvt(odd1, src1, preg, PART_ODD);

        vdiv(evenQuotient, even0, even1, preg, MODE_ZEROING);
        vdiv(oddQuotient, odd0, odd1, preg, MODE_ZEROING);

        vtrc(evenQuotient, evenQuotient, ROUND_F, preg);
        vtrc(oddQuotient, oddQuotient, ROUND_F, preg);

        vmul(evenQuotient, evenQuotient, even1, preg, MODE_ZEROING);
        vmul(oddQuotient, oddQuotient, odd1, preg, MODE_ZEROING);

        vsub(evenQuotient, even0, evenQuotient, preg, MODE_ZEROING);
        vsub(oddQuotient, odd0, oddQuotient, preg, MODE_ZEROING);

        vcvt(evenDst, evenQuotient, preg, ROUND_Z, RS_ENABLE, PART_EVEN);
        vcvt(oddDst, oddQuotient, preg, ROUND_Z, RS_ENABLE, PART_ODD);

        vor(dst, evenDst, oddDst, preg);

        vmul(diffSign, src1, dst, preg, MODE_ZEROING);
        vcmps_lt(diffSignMask, diffSign, 0.0f, preg);
        vadd(dst, dst, src1, diffSignMask, MODE_MERGING);

        vdup((RegTensor<U>&)absReg, abs, preg, MODE_ZEROING);
        vand((RegTensor<U>&)src0Abs, (RegTensor<U>&)src0, (RegTensor<U>&)absReg, preg);
        vcmps_eq(infMask, (RegTensor<U>&)src0Abs, inf, preg);
        vdup((RegTensor<U>&)nanReg, nan, infMask, MODE_ZEROING);
        vsel(dst, nanReg, dst, infMask);
    }
};

} // namespace pto
#endif // TREM_COMMON_REGISTER_HPP
