/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_ISA_COSTMODEL_OP_STRUCT_HPP
#define PTO_ISA_COSTMODEL_OP_STRUCT_HPP

#include <iostream>
#include <vector>
#include "pto/costmodel/costmodel_types.hpp"

#ifndef B16_REPEAT_MAX
#define B16_REPEAT_MAX 65535
#endif

namespace pto {

// BinOp
struct AddOp {
    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats> &stats, uint8_t repeats)
    {
        stats.emplace_back("vadd", repeats);
    }

    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats> &stats, uint8_t repeats, uint8_t dstRepeatStride,
                                      uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        stats.emplace_back("vadd", repeats);
    }
};

struct MulOp {
    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats> &stats, uint8_t repeats)
    {
        stats.emplace_back("vmul", repeats);
    }

    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats> &stats, uint8_t repeats, uint8_t dstRepeatStride,
                                      uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        stats.emplace_back("vmul", repeats);
    }
};

struct SubOp {
    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats> &stats, uint8_t repeats)
    {
        stats.emplace_back("vsub", repeats);
    }

    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats> &stats, uint8_t repeats, uint8_t dstRepeatStride,
                                      uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        stats.emplace_back("vsub", repeats);
    }
};

// BinSOp
struct AddSOp {
    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats> &stats, uint8_t repeats)
    {
        stats.emplace_back("vadds", repeats);
    }

    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats> &stats, uint8_t repeats, uint8_t dstRepeatStride,
                                       uint8_t srcRepeatStride)
    {
        stats.emplace_back("vadds", repeats);
    }
};

struct MulSOp {
    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats> &stats, uint8_t repeats)
    {
        stats.emplace_back("vmuls", repeats);
    }

    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats> &stats, uint8_t repeats, uint8_t dstRepeatStride,
                                       uint8_t srcRepeatStride)
    {
        stats.emplace_back("vmuls", repeats);
    }
};

struct MinSOp {
    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats> &stats, uint8_t repeats)
    {
        stats.emplace_back("vmins", repeats);
    }

    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats> &stats, uint8_t repeats, uint8_t dstRepeatStride,
                                       uint8_t srcRepeatStride)
    {
        stats.emplace_back("vmins", repeats);
    }
};

struct DivSOp {
    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats> &stats, uint8_t rpt)
    {
        stats.emplace_back("vdivs", rpt);
    }

    PTO_INTERNAL static void BinSInstr(std::vector<CostModelStats> &stats, uint8_t rpt, uint8_t dstRepeatStride,
                                       uint8_t srcRepeatStride)
    {
        stats.emplace_back("vdivs", rpt);
    }
};

// UnaryOp
struct AbsOp {
    PTO_INTERNAL static void UnaryInstr(std::vector<CostModelStats> &stats, uint8_t repeats,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        stats.emplace_back("vabs", repeats);
    }
};

struct ExpOp {
    PTO_INTERNAL static void UnaryInstr(std::vector<CostModelStats> &stats, uint8_t repeats,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        stats.emplace_back("vexp", repeats);
    }
};

struct SqrtOp {
    PTO_INTERNAL static void UnaryInstr(std::vector<CostModelStats> &stats, uint8_t repeats,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        stats.emplace_back("vsqrt", repeats);
    }
};

// Reduce Op Backend logistic
template <typename InstrOp>
struct TRowReduceOp {
    PTO_INTERNAL static void BinInstr(std::vector<CostModelStats> &stats, uint8_t rptTimes, uint16_t dstRptStride,
                                      uint16_t src0RptStride, uint16_t src1RptStride)
    {
        InstrOp::BinInstrImpl(stats, rptTimes, dstRptStride, src0RptStride, src1RptStride);
    }

    PTO_INTERNAL static void ReduceInstr(std::vector<CostModelStats> &stats, uint8_t rptTimes, uint16_t dstRptStride,
                                         uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        InstrOp::ReduceInstrImpl(stats, rptTimes, dstRptStride, srcBlkStride, srcRptStride);
    }

    template <int Rows, int ValidRow, int Cols, int ValidCol>
    PTO_INTERNAL static void ReduceOptFP32_64x128(std::vector<CostModelStats> &stats)
    {
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 2, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        InstrOp::BinInstrImpl(stats, ValidRow / 8, 8, 16, 16, 1, 2, 2);
        stats.emplace_back("PIPE_V");
        InstrOp::GroupReduceInstrImpl(stats, ValidRow / 8, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        return;
    }

    template <int Rows, int ValidRow, int Cols, int ValidCol>
    PTO_INTERNAL static void ReduceOptFP32_32x256(std::vector<CostModelStats> &stats)
    {
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 4, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        InstrOp::BinInstrImpl(stats, ValidRow / 4, 8, 16, 16, 1, 2, 2);
        stats.emplace_back("PIPE_V");
        InstrOp::BinInstrImpl(stats, ValidRow / 8, 8, 16, 16, 1, 2, 2);
        stats.emplace_back("PIPE_V");
        InstrOp::GroupReduceInstrImpl(stats, ValidRow / 8, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        return;
    }

    template <int Rows, int ValidRow, int Cols, int ValidCol>
    PTO_INTERNAL static void ReduceOptFP32_16x512(std::vector<CostModelStats> &stats)
    {
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 8, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        InstrOp::GroupReduceInstrImpl(stats, ValidRow, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        InstrOp::GroupReduceInstrImpl(stats, ValidRow / 8, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        return;
    }

    template <int Rows, int ValidRow, int Cols, int ValidCol>
    PTO_INTERNAL static void ReduceOptFP32_8x1024(std::vector<CostModelStats> &stats)
    {
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 16, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        InstrOp::GroupReduceInstrImpl(stats, ValidRow * 2, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        InstrOp::BinInstrImpl(stats, ValidRow / 8, 8, 16, 16, 1, 2, 2);
        stats.emplace_back("PIPE_V");
        InstrOp::GroupReduceInstrImpl(stats, ValidRow / 8, 1, 1, 8);
        stats.emplace_back("PIPE_V");
        return;
    }

    template <bool CntModeEn, int Cols, uint32_t DstStride, uint32_t SrcStride, uint8_t ElemPerRpt>
    PTO_INTERNAL static void ReduceInstrByMode(std::vector<CostModelStats> &stats, unsigned rptTimes)
    {
        if constexpr (DstStride > B16_REPEAT_MAX) {
            for (int i = 0; i < rptTimes; i++) {
                ReduceInstr(stats, 1, 0, 1, 0);
            }
        } else if constexpr (CntModeEn) {
            ReduceInstr(stats, 0, DstStride, 1, SrcStride);
        } else {
            ReduceInstr(stats, rptTimes, DstStride, 1, SrcStride);
        }
    }

    template <bool CntModeEn, int DstCols, int Src0Cols, int Src1Cols, uint32_t DstStride, uint32_t Src0RptStride,
              uint32_t Src1RptStride, uint8_t ElemPerRpt>
    PTO_INTERNAL static void BinInstrByMode(std::vector<CostModelStats> &stats, unsigned rptTimes)
    {
        if constexpr (DstStride > REPEAT_MAX || Src0RptStride > REPEAT_MAX || Src1RptStride > REPEAT_MAX) {
            for (int i = 0; i < rptTimes; i++) {
                BinInstr(stats, 1, 0, 0, 0);
            }
        } else if constexpr (CntModeEn) {
            BinInstr(stats, 0, DstStride, Src0RptStride, Src1RptStride);
        } else {
            BinInstr(stats, rptTimes, DstStride, Src0RptStride, Src1RptStride);
        }
    }

    template <int TmpCols, int SrcCols, uint32_t TmpStride, uint32_t SrcStride, uint8_t ElemPerRpt>
    PTO_INTERNAL static void FillTmp(std::vector<CostModelStats> &stats, int srcRptPerRow, int validRow, int validCol)
    {
        if (validCol >= 2 * ElemPerRpt) {
            BinInstrByMode<true, TmpCols, SrcCols, SrcCols, TmpStride, SrcStride, SrcStride, ElemPerRpt>(stats,
                                                                                                         validRow);
            stats.emplace_back("PIPE_V");
        }
    }

    template <int TmpCols, int SrcCols, uint32_t TmpStride, uint32_t SrcStride, uint8_t ElemPerRpt>
    PTO_INTERNAL static void TmpProc(std::vector<CostModelStats> &stats, int srcRptPerRow, int validRow)
    {
        for (int i = 2; i < srcRptPerRow; ++i) {
            BinInstrByMode<true, TmpCols, TmpCols, SrcCols, TmpStride, TmpStride, SrcStride, ElemPerRpt>(stats,
                                                                                                         validRow);
            stats.emplace_back("PIPE_V");
        }
    }
};

// RowSum Op Backend logistic
struct TRowSumOp : TRowReduceOp<TRowSumOp> {
    PTO_INTERNAL static void BinInstrImpl(std::vector<CostModelStats> &stats, uint8_t rptTimes, uint16_t dstRptStride,
                                          uint16_t src0RptStride, uint16_t src1RptStride, uint8_t dstBlockStride = 1,
                                          uint8_t src0BlockStride = 1, uint8_t src1BlockStride = 1)
    {
        stats.emplace_back("vadd", rptTimes, dstBlockStride, src0BlockStride, src1BlockStride, dstRptStride,
                           src0RptStride, src1RptStride);
    }

    PTO_INTERNAL static void ReduceInstrImpl(std::vector<CostModelStats> &stats, uint8_t rptTimes,
                                             uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        stats.emplace_back("vcadd", rptTimes, dstRptStride, srcBlkStride, srcRptStride, "ONLY_VALUE");
    }

    PTO_INTERNAL static void GroupReduceInstrImpl(std::vector<CostModelStats> &stats, uint8_t rptTimes,
                                                  uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        stats.emplace_back("vcgadd", rptTimes, dstRptStride, srcBlkStride, srcRptStride, "None");
    }
};

// RowMax Op Backend logistic
struct TRowMaxOp : TRowReduceOp<TRowMaxOp> {
    PTO_INTERNAL static void BinInstrImpl(std::vector<CostModelStats> &stats, uint8_t rptTimes, uint16_t dstRptStride,
                                          uint16_t src0RptStride, uint16_t src1RptStride, uint8_t dstBlockStride = 1,
                                          uint8_t src0BlockStride = 1, uint8_t src1BlockStride = 1)
    {
        stats.emplace_back("vmax", rptTimes, dstBlockStride, src0BlockStride, src1BlockStride, dstRptStride,
                           src0RptStride, src1RptStride);
    }

    PTO_INTERNAL static void ReduceInstrImpl(std::vector<CostModelStats> &stats, uint8_t rptTimes,
                                             uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        stats.emplace_back("vcmax", rptTimes, dstRptStride, srcBlkStride, srcRptStride, "ONLY_VALUE");
    }

    PTO_INTERNAL static void GroupReduceInstrImpl(std::vector<CostModelStats> &stats, uint8_t rptTimes,
                                                  uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        stats.emplace_back("vcgmax", rptTimes, dstRptStride, srcBlkStride, srcRptStride, "None");
    }
};

struct TRowMinOp : TRowReduceOp<TRowMinOp> {
    PTO_INTERNAL static void BinInstrImpl(std::vector<CostModelStats> &stats, uint8_t rptTimes, uint16_t dstRptStride,
                                          uint16_t src0RptStride, uint16_t src1RptStride, uint8_t dstBlockStride = 1,
                                          uint8_t src0BlockStride = 1, uint8_t src1BlockStride = 1)
    {
        stats.emplace_back("vmin", rptTimes, dstBlockStride, src0BlockStride, src1BlockStride, dstRptStride,
                           src0RptStride, src1RptStride);
    }

    PTO_INTERNAL static void ReduceInstrImpl(std::vector<CostModelStats> &stats, uint8_t rptTimes,
                                             uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        stats.emplace_back("vcmin", rptTimes, dstRptStride, srcBlkStride, srcRptStride, "ONLY_VALUE");
    }

    PTO_INTERNAL static void GroupReduceInstrImpl(std::vector<CostModelStats> &stats, uint8_t rptTimes,
                                                  uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        stats.emplace_back("vcgmin", rptTimes, dstRptStride, srcBlkStride, srcRptStride, "None");
    }
};

struct COLMAXOp {
    PTO_INTERNAL static void ReduceInstr(std::vector<CostModelStats> &stats, uint8_t repeats, uint8_t dstRepeatStride,
                                         uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        stats.emplace_back("vmax", repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
    }
};

struct COLMINOp {
    PTO_INTERNAL static void ReduceInstr(std::vector<CostModelStats> &stats, uint8_t repeats, uint8_t dstRepeatStride,
                                         uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        stats.emplace_back("vmin", repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
    }
};

template <typename InstrOp>
struct TColReduceOp {
    template <int dupSrcStride>
    PTO_INTERNAL static void ColReduceInstrByMode(std::vector<CostModelStats> &stats, int numRepeatPerLine,
                                                  int numRemainPerLine, int elementsPerLine, int validRow)
    {
        if (numRepeatPerLine > 0) {
            stats.emplace_back("mask", 0, elementsPerLine);
            for (int i = 1; i < validRow; i++) {
                InstrOp::ReduceInstr(stats, 0, 8, 8, 8);
                stats.emplace_back("PIPE_V");
            }
        }

        if (numRemainPerLine > 0) {
            stats.emplace_back("mask", GetContinuousMask1(numRemainPerLine), GetContinuousMask0(numRemainPerLine));
            for (int i = 1; i < validRow; i++) {
                InstrOp::ReduceInstr(stats, 1, 8, 8, 8);
                stats.emplace_back("PIPE_V");
            }
            stats.emplace_back("mask", -1, -1);
        }
    }
};

} // namespace pto

#endif // PTO_ISA_COSTMODEL_OP_STRUCT_HPP
