/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COSTMODEL_TYPES_HPP
#define PTO_COSTMODEL_TYPES_HPP

#include <cstdint>
#include <iostream>

namespace pto {

struct CostModelStats {
    std::string cceInstName;
    int repeats;
    int mask1{};
    int mask0{};
    int dstBlockStride;
    int src0BlockStride;
    int src1BlockStride;
    int dstRepeatStride;
    int src0RepeatStride;
    int src1RepeatStride;
    std::string order; // vcmax/vcmin专用,取值VALUE_INDEX/INDEX_VALUE/ONLY_VALUE/ONLY_INDEX
    bool mode{};       // vcadd专用

    int nBurst{};
    int lenBurst{};
    int leftPaddingNum;
    int rightPaddingNum;
    int convControl;
    int srcGap{};
    int dstGap{};
    int padMode;
    int padConfig; // set_mov_pad_val参数，即存入copy_gm_to_ubuf_align接口的填充值
    int byteMode;

    // mmad专用
    int m{};
    int k{};
    int n{};

    void setCceInstName(std::string cceInstName_)
    {
        cceInstName = cceInstName_;
    }

    // BinOp
    CostModelStats(const std::string cceInstName_, int repeats_, int dstBlockStride_, int src0BlockStride_,
                   int src1BlockStride_, int dstRepeatStride_, int src0RepeatStride_, int src1RepeatStride_)
        : cceInstName(cceInstName_),
          repeats(repeats_),
          dstBlockStride(dstBlockStride_),
          src0BlockStride(src0BlockStride_),
          src1BlockStride(src1BlockStride_),
          dstRepeatStride(dstRepeatStride_),
          src0RepeatStride(src0RepeatStride_),
          src1RepeatStride(src1RepeatStride_)
    {}

    // pipe_barrier
    CostModelStats(const std::string cceInstName_ = "PIPE_V") : cceInstName(cceInstName_)
    {}

    // move
    CostModelStats(const std::string cceInstName_, int nBurst_, int lenBurst_, int srcGap_, int dstGap_)
        : cceInstName(cceInstName_), nBurst(nBurst_), lenBurst(lenBurst_), srcGap(srcGap_), dstGap(dstGap_)
    {}

    // BinSOp UnaryOp
    CostModelStats(const std::string cceInstName_, int repeats_, int dstBlockStride_, int srcBlockStride_,
                   int dstRepeatStride_, int srcRepeatStride_)
        : cceInstName(cceInstName_),
          repeats(repeats_),
          dstBlockStride(dstBlockStride_),
          src0BlockStride(srcBlockStride_),
          dstRepeatStride(dstRepeatStride_),
          src0RepeatStride(srcRepeatStride_)
    {}

    // vcmax/vcmin/vcgadd/vcgmax/vcgmin/vcpadd
    CostModelStats(const std::string cceInstName_, int repeats_, int dstRepeatStride_, int srcBlockStride_,
                   int srcRepeatStride_, const std::string order_)
        : cceInstName(cceInstName_),
          repeats(repeats_),
          dstRepeatStride(dstRepeatStride_),
          src0BlockStride(srcBlockStride_),
          src0RepeatStride(srcRepeatStride_),
          order(order_)
    {}

    // vcadd
    CostModelStats(const std::string cceInstName_, int repeats_, int dstRepeatStride_, int srcBlockStride_,
                   int srcRepeatStride_, bool mode_)
        : cceInstName(cceInstName_),
          repeats(repeats_),
          dstRepeatStride(dstRepeatStride_),
          src0BlockStride(srcBlockStride_),
          src0RepeatStride(srcRepeatStride_),
          mode(mode_)
    {}

    // simple mode
    CostModelStats(const std::string cceInstName_, int repeats_) : cceInstName(cceInstName_), repeats(repeats_)
    {}

    // mask
    CostModelStats(const std::string cceInstName_, int mask1_, int mask0_)
        : cceInstName(cceInstName_), mask1(mask1_), mask0(mask0_)
    {}

    // mmad
    static CostModelStats MakeMmad(const std::string &cceInstName_, int m_, int k_, int n_)
    {
        CostModelStats s(cceInstName_, 1);
        s.m = m_;
        s.k = k_;
        s.n = n_;
        return s;
    }

    void setLeftPaddingNum(int paddingNum_)
    {
        leftPaddingNum = paddingNum_;
    }

    int getLeftPaddingNum()
    {
        return leftPaddingNum;
    }

    void setRightPaddingNum(int paddingNum_)
    {
        rightPaddingNum = paddingNum_;
    }

    int getRightPaddingNum()
    {
        return rightPaddingNum;
    }

    void setPadMode(int mode_)
    {
        padMode = mode_;
    }

    int getPadMode()
    {
        return padMode;
    }

    void setByteMode(int mode_)
    {
        byteMode = mode_;
    }

    int getByteMode()
    {
        return byteMode;
    }

    void setPadConfig(int padConfig_)
    {
        padConfig = padConfig_;
    }

    int getPadConfig()
    {
        return padConfig;
    }

    void setConvControl(int convControl_)
    {
        convControl = convControl_;
    }

    int getConvControl()
    {
        return convControl;
    }
};

struct CostModelParams {
    float startup_cycles = 0.0f;
    float completion_cycles = 0.0f;
    float per_repeat_cycles = 0.0f;
    float interval_cycles = 0.0f;
    float mask_effect = 1.0f;
    float bank_conflict_cycles = 0.0f;
};

inline uint64_t GetContinuousMask1(unsigned n)
{
    return static_cast<uint64_t>(
        (n > MASK_LEN) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n - MASK_LEN)) - 1) : 0);
}

inline uint64_t GetContinuousMask0(unsigned n)
{
    return static_cast<uint64_t>((n >= MASK_LEN) ? 0xffffffffffffffff :
                                                   (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n)) - 1));
}

inline int32_t CeilDivision(int32_t num1, int32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

} // namespace pto

#endif
