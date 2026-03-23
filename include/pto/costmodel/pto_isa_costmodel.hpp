/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_ISA_COST_MODEL_HPP
#define PTO_ISA_COST_MODEL_HPP

#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <type_traits>
#include <pto/common/pto_tile.hpp>
#include "pto/costmodel/op_struct.hpp"
#include "pto/costmodel/costmodel_types.hpp"
#include "pto/costmodel/a2a3/TBinOp.hpp"
#include "pto/costmodel/a2a3/TBinSOp.hpp"
#include "pto/costmodel/a2a3/TUnaryOp.hpp"
#include "pto/costmodel/a2a3/TColReduceOp.hpp"
#include "pto/costmodel/a2a3/TRowReduceOp.hpp"
#include "pto/costmodel/a2a3/TRowExpandOp.hpp"

namespace pto {

// A2A3 vector instruction timing constants (all values in cycles)
// Startup latencies (pipeline fill time from instruction issue to first output)
constexpr float A2A3_STARTUP_REDUCE = 13.0f; // reduce/transcendental ops (vcg*, vc*, vabs, vexp, vsqrt)
constexpr float A2A3_STARTUP_BINARY = 14.0f; // binary, scalar, dup ops

// Completion latencies (pipeline drain time after last output)
constexpr float A2A3_COMPL_DUP = 14.0f;       // vector_dup
constexpr float A2A3_COMPL_INT_BINOP = 17.0f; // int add/sub/min/max and int scalar (vadds/vmins/vdivs int)
constexpr float A2A3_COMPL_INT_MUL = 18.0f;   // int mul/div (vmul/vdivs int)
constexpr float A2A3_COMPL_FP_BINOP = 19.0f;  // fp add/sub; abs (all dtypes); fp scalar; int32 group-reduce
constexpr float A2A3_COMPL_FP_MUL = 20.0f;    // fp mul/div scalar (vmuls/vdivs fp)
constexpr float A2A3_COMPL_FP_CGOP = 21.0f;   // fp16 group/cross-lane reduce (vcg*/vc* fp16)
constexpr float A2A3_COMPL_FP32_EXP = 26.0f;  // fp32 exp
constexpr float A2A3_COMPL_FP32_SQRT = 27.0f; // fp32 sqrt
constexpr float A2A3_COMPL_FP16_EXP = 28.0f;  // fp16 exp
constexpr float A2A3_COMPL_FP16_SQRT = 29.0f; // fp16 sqrt

// Per-repeat throughput (cycles per VL-aligned repeat within one instruction call)
constexpr float A2A3_RPT_1 = 1.0f; // scalar/unary/single-pass ops
constexpr float A2A3_RPT_2 = 2.0f; // binary vector ops
constexpr float A2A3_RPT_4 = 4.0f; // transcendental ops (exp/sqrt fp16)

// Pipeline configuration
constexpr float A2A3_INTERVAL = 18.0f;   // interval cycles between instruction groups
constexpr float A2A3_MASK_EFFECT = 1.0f; // mask penalty multiplier (1.0 = no extra penalty)
constexpr float A2A3_BANK_NONE = 0.0f;   // no bank conflict penalty

enum class DataType
{
    FP16,
    FP32,
    INT8,
    INT16,
    UINT8,
    INT32,
    BF16
};

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
struct TileInfo {
public:
    using DType = Element_;

    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;

    static constexpr int ValidRow = RowValid_;
    static constexpr int ValidCol = ColValid_;

    static constexpr int GetValidRow()
    {
        return ValidRow;
    }

    static constexpr int GetValidCol()
    {
        return ValidCol;
    }

    static constexpr int RowStride = Cols;
    static constexpr int ColStride = Rows;

    float cycle;
    void SetCycle(const float cycle_)
    {
        cycle = cycle_;
    }

    float GetCycle()
    {
        return cycle;
    }
};

struct InstrTypeHash {
    size_t operator()(const std::pair<std::string, DataType> &key) const
    {
        auto hash_instr = std::hash<std::string>()(key.first);
        auto hash_dtype = std::hash<int>()(static_cast<int>(key.second));
        return hash_instr ^ (hash_dtype << 1);
    }
};

class CostModel {
public:
    static CostModel &GetInstance()
    {
        static CostModel instance;
        return instance;
    }

    CostModel(const CostModel &) = delete;
    CostModel &operator=(const CostModel &) = delete;

    void InitDefaultParams()
    {
        SetParam("PIPE_V", DataType::INT16, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("PIPE_V", DataType::INT32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("PIPE_V", DataType::FP16, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("PIPE_V", DataType::FP32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        SetParam("vector_dup", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vector_dup", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vector_dup", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vector_dup", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TADD
        SetParam("vadd", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vadd", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vadd", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vadd", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TMUL
        SetParam("vmul", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_MUL, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmul", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_MUL, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmul", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_MUL, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmul", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_MUL, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TSUB
        SetParam("vsub", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vsub", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vsub", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vsub", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TEXP
        SetParam("vexp", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP16_EXP, A2A3_RPT_4, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vexp", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP32_EXP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TSQRT
        SetParam("vsqrt", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP16_SQRT, A2A3_RPT_4, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vsqrt", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP32_SQRT, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TADDS
        SetParam("vadds", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vadds", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vadds", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vadds", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TABS
        SetParam("vabs", DataType::INT16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vabs", DataType::INT32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vabs", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vabs", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TMINS
        SetParam("vmins", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmins", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmins", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmins", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TMULS
        SetParam("vmuls", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_MUL, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmuls", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_MUL, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmuls", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_MUL, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmuls", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_MUL, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TDIVS
        SetParam("vdivs", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_MUL, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vdivs", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_MUL, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vdivs", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_MUL, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vdivs", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_MUL, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // vmax
        SetParam("vmax", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmax", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmax", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmax", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // vmin
        SetParam("vmin", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmin", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmin", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmin", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // vcgmax
        SetParam("vcgmax", DataType::INT16, A2A3_STARTUP_REDUCE, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcgmax", DataType::INT32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcgmax", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_CGOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcgmax", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // vcgmin
        SetParam("vcgmin", DataType::INT16, A2A3_STARTUP_REDUCE, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcgmin", DataType::INT32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcgmin", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_CGOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcgmin", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // vcgadd
        SetParam("vcgadd", DataType::INT16, A2A3_STARTUP_REDUCE, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcgadd", DataType::INT32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcgadd", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_CGOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcgadd", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // vcmax
        SetParam("vcmax", DataType::INT16, A2A3_STARTUP_REDUCE, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcmax", DataType::INT32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcmax", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_CGOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcmax", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // vcmin
        SetParam("vcmin", DataType::INT16, A2A3_STARTUP_REDUCE, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcmin", DataType::INT32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcmin", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_CGOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcmin", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // vcadd
        SetParam("vcadd", DataType::INT16, A2A3_STARTUP_REDUCE, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcadd", DataType::INT32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcadd", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_CGOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vcadd", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // copy_ubuf_to_ubuf (memory copy, 0-cycle placeholder)
        SetParam("copy_ubuf_to_ubuf", DataType::INT16, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("copy_ubuf_to_ubuf", DataType::INT32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("copy_ubuf_to_ubuf", DataType::FP16, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("copy_ubuf_to_ubuf", DataType::FP32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // mask (mask set instruction, 0-cycle placeholder)
        SetParam("mask", DataType::INT16, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mask", DataType::INT32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mask", DataType::FP16, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mask", DataType::FP32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
    }

    // TBinOp
    template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
    void BinOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runBinaryOp<Op>(dst, src0, src1);
        dst.SetCycle(PredictCycle<T>(stats));
    }

    // TBinSOp
    template <typename Op, typename TileDataDst, typename TileDataSrc>
    void BinSOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runBinaryScalarOp<Op>(dst, src);
        dst.SetCycle(PredictCycle<T>(stats));
    }

    // TUnaryOp
    template <typename Op, typename TileDataDst, typename TileDataSrc>
    void UnaryOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runUnaryOp<Op>(dst, src);
        dst.SetCycle(PredictCycle<T>(stats));
    }

    // TRowReduceOpPredict
    template <typename Op, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
    void RowReduceOpPredictCycle(const std::string &instr_name, TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
    {
        using T = typename TileDataIn::DType;
        std::vector<CostModelStats> stats =
            runRowReduceOps<T, Op, TileDataOut, TileDataIn, TileDataTmp>(instr_name, dst, src, tmp);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    // TColMax / TColMin
    template <typename Op, typename TileDataOut, typename TileDataIn>
    void ColReduceOpPredictCycle(const std::string &instr_name, TileDataOut &dst, TileDataIn &src)
    {
        using T = typename TileDataIn::DType;
        std::vector<CostModelStats> stats = runColReduceOps<T, Op, TileDataOut, TileDataIn>(dst, src);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    // TColSum
    template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
    void ColSumOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp,
                              bool IsBinary)
    {
        using T = typename TileDataSrc::DType;
        std::vector<CostModelStats> stats =
            runColSumOp<T, TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp, IsBinary);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    // TRowExpand
    template <typename TileDataDst, typename TileDataSrc>
    void RowExpandPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runRowExpandOp<TileDataDst, TileDataSrc>(dst, src);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    template <typename T>
    float PredictCycle(const std::vector<CostModelStats> &stats)
    {
        float total_cycles = 0.0f;
        // first: next real instruction starts a new pipeline segment (pays startup_cycles once)
        // pipe:  previous instruction was a PIPE_V barrier (next real instruction pays interval_cycles)
        bool first = true;
        bool pipe = false;

        for (const auto &stat : stats) {
            const std::string &instr_name = stat.cceInstName;
            DataType dtype = GetDataTypeEnum<T>();
            auto key = std::make_pair(instr_name, dtype);

            if (instr_name == "PIPE_V") {
                pipe = true;
                continue;
            }

            if (!CheckParamExist(key)) {
                fprintf(stderr, "[CostModel] Error: unknown instruction <%s> for dtype <%d>\n", instr_name.c_str(),
                        static_cast<int>(dtype));
                return 0.0f;
            }

            const CostModelParams &params = params_map_.at(key);

            // Startup: paid once for the very first instruction in the sequence
            if (first) {
                total_cycles += params.startup_cycles;
                first = false;
            }

            // Interval: paid for any instruction that immediately follows a PIPE_V barrier
            if (pipe) {
                total_cycles += params.interval_cycles;
                pipe = false;
            }
            total_cycles += stat.repeats * params.per_repeat_cycles;
        }
        fprintf(stdout, "[CostModel] PredictCycle: %.1f\n", total_cycles);
        return total_cycles;
    }

private:
    CostModel()
    {
        InitDefaultParams();
    }

    void SetParam(const std::string &instr_name, DataType dtype, float head, float complete, float computing,
                  float interval, float mask, float bank_conflict)
    {
        params_map_[std::make_pair(instr_name, dtype)] =
            CostModelParams{head, complete, computing, interval, mask, bank_conflict};
    }

    bool CheckParamExist(const std::pair<std::string, DataType> &key)
    {
        return params_map_.count(key) > 0;
    }

    template <typename T>
    DataType GetDataTypeEnum()
    {
        if constexpr (std::is_same_v<T, __bf16>) {
            return DataType::BF16;
        } else if constexpr (std::is_same_v<T, half>) {
            return DataType::FP16;
        } else if constexpr (std::is_same_v<T, float>) {
            return DataType::FP32;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return DataType::INT16;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return DataType::INT32;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return DataType::INT8;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return DataType::UINT8;
        } else {
            return DataType::FP16;
        }
    }

    std::unordered_map<std::pair<std::string, DataType>, CostModelParams, InstrTypeHash> params_map_;
};

} // namespace pto

#endif
