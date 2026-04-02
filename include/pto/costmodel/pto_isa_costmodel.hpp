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
#include "pto/costmodel/a2a3/TLoadOp.hpp"
#include "pto/costmodel/a2a3/TMovOp.hpp"
#include "pto/costmodel/a2a3/TExtractOp.hpp"
#include "pto/costmodel/a2a3/TScatterOp.hpp"
#include "pto/costmodel/a2a3/TMatmulOp.hpp"
#include "pto/costmodel/a2a3/TTransOp.hpp"
#include "pto/costmodel/a2a3/TMrgSortOp.hpp"
#include "pto/costmodel/a2a3/TSelOp.hpp"
#include "pto/costmodel/a2a3/TCvtOp.hpp"
#include "pto/costmodel/a2a3/TSort32Op.hpp"
#include "pto/costmodel/a2a3/TMatmulOp.hpp"

namespace pto {

// A2A3 vector instruction timing constants (all values in cycles)
// Startup latencies (pipeline fill time from instruction issue to first output)
constexpr float A2A3_STARTUP_REDUCE = 13.0f; // reduce/transcendental ops (vcg*, vc*, vabs, vexp, vsqrt)
constexpr float A2A3_STARTUP_BINARY = 14.0f; // binary, scalar, dup ops

// Completion latencies (pipeline drain time after last output)
constexpr float A2A3_COMPL_DUP_VCOPY = 14.0f; // vector_dup
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
constexpr float A2A3_RPT_6 = 6.0f; // merge_sort op

// Pipeline configuration
constexpr float A2A3_INTERVAL = 18.0f;       // interval cycles between instruction groups
constexpr float A2A3_INTERVAL_VCOPY = 13.0f; // interval cycles between instruction groups
constexpr float A2A3_MASK_EFFECT = 1.0f;     // mask penalty multiplier (1.0 = no extra penalty)
constexpr float A2A3_BANK_NONE = 0.0f;       // no bank conflict penalty

// A2A3 data transfer bandwidth constants (B/Cycle), named as SRC_DST using TileType names
constexpr float A2A3_BW_GM_VEC = 128.0f;
constexpr float A2A3_BW_VEC_VEC = 128.0f;
constexpr float A2A3_BW_GM_MAT = 256.0f;
constexpr float A2A3_BW_MAT_LEFT = 256.0f;
constexpr float A2A3_BW_MAT_RIGHT = 128.0f;
constexpr float A2A3_BW_MAT_BIAS = 128.0f;
constexpr float A2A3_BW_MAT_SCALING = 128.0f;
constexpr float A2A3_BW_ACC_MAT = 128.0f;
constexpr float A2A3_BW_MAT_MAT = 32.0f; // l12l1 (TEXTRACT Mat→Mat)

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

constexpr int getDataTypeBytes(DataType type)
{
    switch (type) {
        case DataType::FP16:
            return 2;
        case DataType::FP32:
            return 4;
        case DataType::INT8:
            return 1;
        case DataType::INT16:
            return 2;
        case DataType::UINT8:
            return 1;
        case DataType::INT32:
            return 4;
        case DataType::BF16:
            return 2;
        default:
            return 0;
    }
}

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

    [[nodiscard]] float GetCycle() const
    {
        return cycle;
    }
};

struct InstrTypeHash {
    size_t operator()(const std::pair<std::string, DataType> &key) const
    {
        const auto hash_instr = std::hash<std::string>()(key.first);
        const auto hash_dtype = std::hash<int>()(static_cast<int>(key.second));
        return hash_instr ^ (hash_dtype << 1);
    }
};

struct IntIntHash {
    size_t operator()(const std::pair<int, int> &key) const
    {
        const auto hash_instr = std::hash<int>()(key.first);
        const auto hash_dtype = std::hash<int>()(key.second);
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

        SetParam("vector_dup", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vector_dup", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vector_dup", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vector_dup", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1, A2A3_INTERVAL,
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
        SetParam("copy_ubuf_to_ubuf", DataType::INT16, A2A3_STARTUP_REDUCE, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1,
                 A2A3_INTERVAL_VCOPY, A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("copy_ubuf_to_ubuf", DataType::INT32, A2A3_STARTUP_REDUCE, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1,
                 A2A3_INTERVAL_VCOPY, A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("copy_ubuf_to_ubuf", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1,
                 A2A3_INTERVAL_VCOPY, A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("copy_ubuf_to_ubuf", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1,
                 A2A3_INTERVAL_VCOPY, A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // mask (mask set instruction, 0-cycle placeholder)
        SetParam("mask", DataType::INT16, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mask", DataType::INT32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mask", DataType::FP16, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mask", DataType::FP32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TSEL: vsel (element-wise select, 1 cycle/repeat, startup=14)
        SetParam("vsel", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vsel", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vsel", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vsel", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TCVT: vconv (type conversion, startup=13 like reduce ops)
        SetParam("vconv", DataType::FP16, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vconv", DataType::FP32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vconv", DataType::INT16, A2A3_STARTUP_REDUCE, A2A3_COMPL_INT_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vconv", DataType::INT32, A2A3_STARTUP_REDUCE, A2A3_COMPL_FP_BINOP, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TSORT32: vbitsort (bitonic sort, 2 cycles/repeat)
        SetParam("vbitsort", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_4, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vbitsort", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_4, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vbitsort", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_4, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vbitsort", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_4, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TMRGSORT: vmrgsort4 (merge sort, 2 cycles/repeat)
        SetParam("vmrgsort4", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_6, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("vmrgsort4", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_6, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        SetParam("scatter", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("scatter", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("scatter", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_1, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("scatter", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_DUP_VCOPY, A2A3_RPT_2, A2A3_INTERVAL,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TMATMUL / TGEMV: mmad (PIPE_M cube pipeline — recorded, cycle model TBD)
        SetParam("mmad", DataType::FP32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mmad", DataType::INT32, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // TTRANS: scatter_vnchwconv_b8/b16/b32 (vector-pipeline layout-transpose, startup=14, 2 cycles/repeat)
        // B16 types (sizeof=2): half, int16
        SetParam("scatter_vnchwconv", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_2,
                 A2A3_INTERVAL, A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("scatter_vnchwconv", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2,
                 A2A3_INTERVAL, A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        // B32 types (sizeof=4): float, int32
        SetParam("scatter_vnchwconv", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_COMPL_FP_BINOP, A2A3_RPT_2,
                 A2A3_INTERVAL, A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("scatter_vnchwconv", DataType::INT32, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2,
                 A2A3_INTERVAL, A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        // B8 types (sizeof=1): int8, uint8
        SetParam("scatter_vnchwconv", DataType::INT8, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2,
                 A2A3_INTERVAL, A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("scatter_vnchwconv", DataType::UINT8, A2A3_STARTUP_BINARY, A2A3_COMPL_INT_BINOP, A2A3_RPT_2,
                 A2A3_INTERVAL, A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // mmad
        SetParam("mad", DataType::INT16, A2A3_STARTUP_BINARY, A2A3_BANK_NONE, A2A3_RPT_1, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mad", DataType::INT8, A2A3_STARTUP_BINARY, A2A3_BANK_NONE, A2A3_RPT_1, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mad", DataType::FP16, A2A3_STARTUP_BINARY, A2A3_BANK_NONE, A2A3_RPT_1, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);
        SetParam("mad", DataType::FP32, A2A3_STARTUP_BINARY, A2A3_BANK_NONE, A2A3_RPT_2, A2A3_BANK_NONE,
                 A2A3_MASK_EFFECT, A2A3_BANK_NONE);

        // gm2ub
        SetParam(-1, static_cast<int>(TileType::Vec), A2A3_BW_GM_VEC);
        // ub2ub
        SetParam(static_cast<int>(TileType::Vec), static_cast<int>(TileType::Vec), A2A3_BW_VEC_VEC);
        // gm2l1
        SetParam(-1, static_cast<int>(TileType::Mat), A2A3_BW_GM_MAT);
        // l12l0A
        SetParam(static_cast<int>(TileType::Mat), static_cast<int>(TileType::Left), A2A3_BW_MAT_LEFT);
        // l12l0B
        SetParam(static_cast<int>(TileType::Mat), static_cast<int>(TileType::Right), A2A3_BW_MAT_RIGHT);
        // l12BT
        SetParam(static_cast<int>(TileType::Mat), static_cast<int>(TileType::Bias), A2A3_BW_MAT_BIAS);
        // l12FP
        SetParam(static_cast<int>(TileType::Mat), static_cast<int>(TileType::Scaling), A2A3_BW_MAT_SCALING);
        // l0C2l1
        SetParam(static_cast<int>(TileType::Acc), static_cast<int>(TileType::Mat), A2A3_BW_ACC_MAT);
        // l12l1 (TEXTRACT Mat→Mat)
        SetParam(static_cast<int>(TileType::Mat), static_cast<int>(TileType::Mat), A2A3_BW_MAT_MAT);
    }

    // TBinOp
    template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
    void BinOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runBinaryOp<Op>(dst, src0, src1);
        dst.SetCycle(VecInstPredictCycle<T>(stats));
    }

    // TBinSOp
    template <typename Op, typename TileDataDst, typename TileDataSrc>
    void BinSOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runBinaryScalarOp<Op>(dst, src);
        dst.SetCycle(VecInstPredictCycle<T>(stats));
    }

    // TUnaryOp
    template <typename Op, typename TileDataDst, typename TileDataSrc>
    void UnaryOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runUnaryOp<Op>(dst, src);
        dst.SetCycle(VecInstPredictCycle<T>(stats));
    }

    // TRowReduceOpPredict
    template <typename Op, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
    void RowReduceOpPredictCycle(const std::string &instr_name, TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
    {
        using T = typename TileDataIn::DType;
        std::vector<CostModelStats> stats =
            runRowReduceOps<T, Op, TileDataOut, TileDataIn, TileDataTmp>(instr_name, dst, src, tmp);
        float totalCycles = VecInstPredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    // TColMax / TColMin
    template <typename Op, typename TileDataOut, typename TileDataIn>
    void ColReduceOpPredictCycle(const std::string &instr_name, TileDataOut &dst, TileDataIn &src)
    {
        using T = TileDataIn::DType;
        std::vector<CostModelStats> stats = runColReduceOps<T, Op, TileDataOut, TileDataIn>(dst, src);
        float totalCycles = VecInstPredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    // TColSum
    template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
    void ColSumOpPredictCycle(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, const bool IsBinary)
    {
        using T = TileDataSrc::DType;
        const std::vector<CostModelStats> stats =
            runColSumOp<T, TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp, IsBinary);
        float totalCycles = VecInstPredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    // TRowExpand
    template <typename TileDataDst, typename TileDataSrc>
    void RowExpandPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = TileDataDst::DType;
        std::vector<CostModelStats> stats = runRowExpandOp<TileDataDst, TileDataSrc>(dst, src);
        float totalCycles = VecInstPredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    template <typename TileRes, typename TileLeft, typename TileRight>
    void MatmulPredictCycle(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
    {
        using T = typename TileLeft::DType;
        auto stats = runMatmulOp(aMatrix, bMatrix);
        float total_cycles = CubeInstPredictCycle<T>(stats);
        cMatrix.SetCycle(total_cycles);
    }

    template <typename TileRes, typename TileLeft, typename TileRight>
    void MatmulBiasPredictCycle(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
    {
        using T = typename TileLeft::DType;
        auto stats = runMatmulBiasOp(aMatrix, bMatrix);
        float total_cycles = CubeInstPredictCycle<T>(stats);
        cMatrix.SetCycle(total_cycles);
    }

    // TCvt
    template <typename TileDataD, typename TileDataS>
    void CvtOpPredictCycle(const std::string &instr_name, TileDataD &dst, TileDataS &src, RoundMode mode,
                           SaturationMode satMode)
    {
        using T = typename TileDataD::DType; // conv的类型需单独考虑
        std::vector<CostModelStats> stats;
        runTCvtOp<TileDataD, TileDataS>(stats, dst, src, mode, satMode);
        float totalCycles = VecInstPredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    // TSel
    template <typename DstTile, typename MaskTile, typename Src0Tile, typename Src1Tile, typename TmpTile>
    void SelOpPredictCycle(const std::string &instr_name, DstTile &dst)
    {
        using T = typename DstTile::DType;
        std::vector<CostModelStats> stats = runTSelOp<DstTile, MaskTile, Src0Tile, Src1Tile, TmpTile>(dst);
        float totalCycles = VecInstPredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
    }

    // TMov
    template <typename DstTileData, typename SrcTileData>
    void MovOpPredictCycle(const std::string &instr_name, DstTileData &dst, SrcTileData &src)
    {
        using T = typename DstTileData::DType;
        std::vector<CostModelStats> stats;
        runTMovOp<DstTileData, SrcTileData>(stats, dst, src);
        float totalCycles = DataTransInstPredictCycle<T, DstTileData, SrcTileData>(stats, dst, src);
        dst.SetCycle(totalCycles);
    }

    // TMov with mode
    template <typename DstTileData, typename SrcTileData, QuantMode_t quantPre, ReluPreMode reluMode>
    void MovModeOpPredictCycle(const std::string &instr_name, DstTileData &dst, SrcTileData &src)
    {
        using T = typename DstTileData::DType;
        uint16_t m = src.GetValidRow();
        uint16_t n = src.GetValidCol();
        std::vector<CostModelStats> stats;
        TMovCcToCb<DstTileData, SrcTileData, quantPre, reluMode>(stats, m, n);
        float totalCycles = DataTransInstPredictCycle<T, DstTileData, SrcTileData>(stats, dst, src);
        dst.SetCycle(totalCycles);
    }

    // TLoad
    template <typename TileData, typename GlobalData>
    void LoadOpPredictCycle(const std::string &instr_name, TileData &dst, GlobalData &src)
    {
        using T = typename TileData::DType;
        std::vector<CostModelStats> stats = runTLoadOp<TileData, GlobalData>(dst, src);
        float totalCycles = DataTransInstPredictCycle<T, TileData>(stats, dst);
        dst.SetCycle(totalCycles);
    }

    // TExtract
    template <typename DstTileData, typename SrcTileData>
    void ExtractOpPredictCycle(const std::string &instr_name, DstTileData &dst, SrcTileData &src, uint16_t indexRow,
                               uint16_t indexCol)
    {
        using T = typename DstTileData::DType;
        std::vector<CostModelStats> stats;
        runTExtractOp<DstTileData, SrcTileData>(stats, dst, src, indexRow, indexCol);
        float totalCycles = DataTransInstPredictCycle<T, DstTileData, SrcTileData>(stats, dst, src);
        dst.SetCycle(totalCycles);
    }

    // TExtract with mode
    template <typename DstTileData, typename SrcTileData, QuantMode_t quantPre, ReluPreMode reluMode>
    void ExtractModeOpPredictCycle(const std::string &instr_name, DstTileData &dst, SrcTileData &src, uint16_t indexRow,
                                   uint16_t indexCol)
    {
        using T = typename DstTileData::DType;
        std::vector<CostModelStats> stats;
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(stats, dst.GetValidRow(), dst.GetValidCol(),
                                                                       indexRow, indexCol);
        float totalCycles = DataTransInstPredictCycle<T, DstTileData, SrcTileData>(stats, dst, src);
        dst.SetCycle(totalCycles);
    }

    template <typename T>
    [[nodiscard]] float VecInstPredictCycle(const std::vector<CostModelStats> &stats) const
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

            if (instr_name == "PIPE_V" || instr_name == "pipe_barrier") {
                pipe = true;
                continue;
            }

            if (!CheckParamExist(key)) {
                fprintf(stderr, "[CostModel] Error: unknown instruction <%s> for dtype <%d>\n", instr_name.c_str(),
                        static_cast<int>(dtype));
                return 0.0f;
            }

            const CostModelParams &vec_params = params_map_.at(key);

            // Startup: paid once for the very first instruction in the sequence
            if (first) {
                total_cycles += vec_params.startup_cycles;
                first = false;
            }

            // Interval: paid for any instruction that immediately follows a PIPE_V barrier
            if (pipe) {
                total_cycles += vec_params.interval_cycles;
                pipe = false;
            }
            total_cycles += stat.repeats * vec_params.per_repeat_cycles;
        }

        return total_cycles;
    }

    template <typename T>
    [[nodiscard]] float CubeInstPredictCycle(const std::vector<CostModelStats> &stats) const
    {
        float total_cycles = 0.0f;
        bool first = true;
        for (const auto &stat : stats) {
            const std::string &instr_name = stat.cceInstName;
            DataType dtype = GetDataTypeEnum<T>();
            auto key = std::make_pair(instr_name, dtype);
            if (!CheckParamExist(key)) {
                fprintf(stderr, "[CostModel] Error: unknown instruction <%s> for dtype <%d>\n", instr_name.c_str(),
                        static_cast<int>(dtype));
                return 0.0f;
            }
            const CostModelParams &cube_params = params_map_.at(key);

            if (first) {
                total_cycles += cube_params.startup_cycles;
                first = false;
            }

            const int baskK = 32 / getDataTypeBytes(dtype);
            const int repeats = (stat.m + 15) / 16 * ((stat.n + 15) / 16) * ((stat.k + baskK - 1) / baskK);
            total_cycles += repeats * cube_params.per_repeat_cycles;
        }

        return total_cycles;
    }

    // TLoad专用
    template <typename T, typename DstTileData>
    [[nodiscard]] float DataTransInstPredictCycle(const std::vector<CostModelStats> &stats, DstTileData &dst)
    {
        int srcType = -1;
        int dstType = -1;

        // gm2ub
        if constexpr (DstTileData::Loc == TileType::Vec) {
            dstType = static_cast<int>(TileType::Vec);
        } else if constexpr (DstTileData::Loc == TileType::Mat) { // gm2l1
            dstType = static_cast<int>(TileType::Mat);
        }

        if constexpr (is_conv_tile_v<DstTileData>) {
            return DataTransInstPredictCycle(srcType, dstType, DstTileData::bufferSize);
        } else {
            return DataTransInstPredictCycle(srcType, dstType, dst.GetValidRow() * dst.GetValidCol() * sizeof(T));
        }
    }

    // TMov/TExtract
    template <typename T, typename DstTileData, typename SrcTileData>
    [[nodiscard]] float DataTransInstPredictCycle(const std::vector<CostModelStats> &stats, DstTileData &dst,
                                                  SrcTileData &src)
    {
        int srcType = -1;
        int dstType = -1;
        if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Left) {
            srcType = static_cast<int>(TileType::Mat);
            dstType = static_cast<int>(TileType::Left);
        } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Right) {
            srcType = static_cast<int>(TileType::Mat);
            dstType = static_cast<int>(TileType::Right);
        } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Bias) {
            srcType = static_cast<int>(TileType::Mat);
            dstType = static_cast<int>(TileType::Bias);
        } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Scaling) {
            srcType = static_cast<int>(TileType::Mat);
            dstType = static_cast<int>(TileType::Scaling);
        } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Vec) {
            srcType = static_cast<int>(TileType::Vec);
            dstType = static_cast<int>(TileType::Vec);
        } else if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
            srcType = static_cast<int>(TileType::Acc);
            dstType = static_cast<int>(TileType::Mat);
        } else if constexpr (SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Mat) {
            srcType = static_cast<int>(TileType::Mat);
            dstType = static_cast<int>(TileType::Mat);
        }

        if constexpr (is_conv_tile_v<SrcTileData>) {
            return DataTransInstPredictCycle(srcType, dstType, DstTileData::bufferSize);
        } else {
            return DataTransInstPredictCycle(srcType, dstType, src.GetValidRow() * src.GetValidCol() * sizeof(T));
        }
    }

    [[nodiscard]] float DataTransInstPredictCycle(int srcType, int dstType, uint32_t bufferSize)
    {
        float total_cycles = 0.0f;
        auto key = std::make_pair(srcType, dstType);
        if (!data_trans_params_map_.contains(key)) {
            fprintf(stderr, "[CostModel] Error: unknown data transfer instruction, srcType: <%d>,  dstType: <%d>\n",
                    srcType, dstType);
            return total_cycles;
        }

        float bandWidth = data_trans_params_map_.at(key);
        if (bandWidth <= 0) {
            total_cycles = 0.0;
            fprintf(stderr, "[CostModel] Error: bandWidth should be larger than 0");
        } else {
            total_cycles = static_cast<int>(bufferSize / bandWidth);
        }
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

    void SetParam(int srcType, int dstType, float bandWidth)
    {
        data_trans_params_map_[std::make_pair(srcType, dstType)] = bandWidth;
    }

    [[nodiscard]] bool CheckParamExist(const std::pair<std::string, DataType> &key) const
    {
        return params_map_.contains(key);
    }

    template <typename T>
    DataType GetDataTypeEnum() const
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
    std::unordered_map<std::pair<int, int>, float, IntIntHash> data_trans_params_map_;
};

} // namespace pto

#endif
