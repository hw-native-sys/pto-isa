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

#include <string>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <type_traits>
#include <pto/common/pto_tile.hpp>
#include "pto/costmodel/costmodel_types.hpp"
#include "pto/costmodel/a2a3/TBinOp.hpp"
#include "pto/costmodel/a2a3/TBinSOp.hpp"
#include "pto/costmodel/a2a3/TUnaryOp.hpp"

namespace pto {
constexpr float HEAD_CYCLE_13 = 13.0;
constexpr float HEAD_CYCLE_14 = 14.0;
constexpr float COMPLETE_CYCLE_17 = 17.0;
constexpr float COMPLETE_CYCLE_18 = 18.0;
constexpr float COMPLETE_CYCLE_19 = 19.0;
constexpr float COMPLETE_CYCLE_20 = 20.0;
constexpr float COMPLETE_CYCLE_26 = 26.0;
constexpr float COMPLETE_CYCLE_27 = 27.0;
constexpr float COMPLETE_CYCLE_28 = 28.0;
constexpr float COMPLETE_CYCLE_29 = 29.0;
constexpr float COMPUTING_CYCLE_1 = 1.0;
constexpr float COMPUTING_CYCLE_2 = 2.0;
constexpr float COMPUTING_CYCLE_4 = 4.0;
constexpr float INTERVAL_CYCLE_18 = 18.0;
constexpr float MASK_1 = 1.0;
constexpr float BANK_CONFLICT_0 = 0.0;

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
        // TADD
        SetParam("TADD", DataType::INT16, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TADD", DataType::INT32, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TADD", DataType::FP16, HEAD_CYCLE_14, COMPLETE_CYCLE_19, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);
        SetParam("TADD", DataType::FP32, HEAD_CYCLE_14, COMPLETE_CYCLE_19, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);

        // TMUL
        SetParam("TMUL", DataType::INT16, HEAD_CYCLE_14, COMPLETE_CYCLE_18, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TMUL", DataType::INT32, HEAD_CYCLE_14, COMPLETE_CYCLE_18, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TMUL", DataType::FP16, HEAD_CYCLE_14, COMPLETE_CYCLE_20, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);
        SetParam("TMUL", DataType::FP32, HEAD_CYCLE_14, COMPLETE_CYCLE_20, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);

        // TSUB
        SetParam("TSUB", DataType::INT16, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TSUB", DataType::INT32, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TSUB", DataType::FP16, HEAD_CYCLE_14, COMPLETE_CYCLE_19, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);
        SetParam("TSUB", DataType::FP32, HEAD_CYCLE_14, COMPLETE_CYCLE_19, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);

        // TEXP
        SetParam("TEXP", DataType::FP16, HEAD_CYCLE_13, COMPLETE_CYCLE_28, COMPUTING_CYCLE_4, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);
        SetParam("TEXP", DataType::FP32, HEAD_CYCLE_13, COMPLETE_CYCLE_26, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);

        // TSQRT
        SetParam("TSQRT", DataType::FP16, HEAD_CYCLE_13, COMPLETE_CYCLE_29, COMPUTING_CYCLE_4, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TSQRT", DataType::FP32, HEAD_CYCLE_13, COMPLETE_CYCLE_27, COMPUTING_CYCLE_2, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);

        // TADDS
        SetParam("TADDS", DataType::INT16, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TADDS", DataType::INT32, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TADDS", DataType::FP16, HEAD_CYCLE_14, COMPLETE_CYCLE_19, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TADDS", DataType::FP32, HEAD_CYCLE_14, COMPLETE_CYCLE_19, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);

        // TABS
        SetParam("TABS", DataType::INT16, HEAD_CYCLE_13, COMPLETE_CYCLE_19, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TABS", DataType::INT32, HEAD_CYCLE_13, COMPLETE_CYCLE_19, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TABS", DataType::FP16, HEAD_CYCLE_13, COMPLETE_CYCLE_19, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);
        SetParam("TABS", DataType::FP32, HEAD_CYCLE_13, COMPLETE_CYCLE_19, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18, MASK_1,
                 BANK_CONFLICT_0);

        // TMINS
        SetParam("TMINS", DataType::INT16, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TMINS", DataType::INT32, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TMINS", DataType::FP16, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TMINS", DataType::FP32, HEAD_CYCLE_14, COMPLETE_CYCLE_17, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);

        // TMULS
        SetParam("TMULS", DataType::INT16, HEAD_CYCLE_14, COMPLETE_CYCLE_18, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TMULS", DataType::INT32, HEAD_CYCLE_14, COMPLETE_CYCLE_18, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TMULS", DataType::FP16, HEAD_CYCLE_14, COMPLETE_CYCLE_20, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TMULS", DataType::FP32, HEAD_CYCLE_14, COMPLETE_CYCLE_20, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);

        // TDIVS
        SetParam("TDIVS", DataType::INT16, HEAD_CYCLE_14, COMPLETE_CYCLE_18, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TDIVS", DataType::INT32, HEAD_CYCLE_14, COMPLETE_CYCLE_18, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TDIVS", DataType::FP16, HEAD_CYCLE_14, COMPLETE_CYCLE_20, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
        SetParam("TDIVS", DataType::FP32, HEAD_CYCLE_14, COMPLETE_CYCLE_20, COMPUTING_CYCLE_1, INTERVAL_CYCLE_18,
                 MASK_1, BANK_CONFLICT_0);
    }

    // TBinOp
    template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
    void BinOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
    {
        using T = typename TileDataDst::DType;
        CostModelStats stats = runBinaryOp(dst, src0, src1);
        float resultCycles = PredictCycle<T>(instr_name, stats);
        dst.SetCycle(resultCycles);
    }

    // TBinSOp
    template <typename TileDataDst, typename TileDataSrc>
    void BinSOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src,
                            TileDataSrc::DType scalar)
    {
        using T = typename TileDataSrc::DType;
        CostModelStats stats = runBinaryScalarOp(dst, src);
        float resultCycles = PredictCycle<T>(instr_name, stats);
        dst.SetCycle(resultCycles);
    }

    // TUnaryOp
    template <typename TileDataDst, typename TileDataSrc>
    void UnaryOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = typename TileDataDst::DType;
        CostModelStats stats = runUnaryOp(dst, src);
        float resultCycles = PredictCycle<T>(instr_name, stats);
        dst.SetCycle(resultCycles);
    }

    template <typename T>
    float PredictCycle(const std::string &instr_name, const CostModelStats &stats)
    {
        DataType dtype = GetDataTypeEnum<T>();
        auto key = std::make_pair(instr_name, dtype);
        if (!CheckParamExist(key)) {
            fprintf(stderr, "[CostModel] Error: <%s> <%d> \n", instr_name.c_str(), static_cast<int>(dtype));
            return 0.0f;
        }

        const CostModelParams &params = params_map_.at(key);
        float masked_repeat_penalty = params.per_repeat_cycles * (params.mask_effect - 1.0f);
        int effective_repeats = stats.total_repeats > 0 ? stats.total_repeats - 1 : 0;

        float sum_cycles = params.startup_cycles + params.completion_cycles +
                           (effective_repeats * params.per_repeat_cycles) +
                           (stats.masked_repeats * masked_repeat_penalty) + params.bank_conflict_cycles;

        return sum_cycles;
    }

private:
    CostModel()
    {
        InitDefaultParams();
    }

    void SetParam(const std::string &instr_name, DataType dtype, double head, double complete, double computing,
                  double interval, double mask, double bank_conflict)
    {
        auto key = std::make_pair(instr_name, dtype);
        params_map_[key] = CostModelParams{
            static_cast<float>(head),     static_cast<float>(complete), static_cast<float>(computing),
            static_cast<float>(interval), static_cast<float>(mask),     static_cast<float>(bank_conflict),
        };
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
            fprintf(stderr, "[CostModel] Warning: unknow data type, use FP16 instead.\n");
            return DataType::FP16;
        }
    }

    std::unordered_map<std::pair<std::string, DataType>, CostModelParams, InstrTypeHash> params_map_;
};

} // namespace pto

#endif
