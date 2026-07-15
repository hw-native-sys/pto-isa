/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_SOFTMAX_HPP
#define PTO_MACRO_FA_SOFTMAX_HPP

#include <pto/pto-inst.hpp>

namespace pto {

constexpr PTO_INTERNAL float constexpr_sqrt(float x)
{
    if (x <= 0.0f)
        return 0.0f;
    float guess = x;
    for (int i = 0; i < 8; ++i) {
        guess = 0.5f * (guess + x / guess);
    }
    return guess;
}

constexpr PTO_INTERNAL float constexpr_inv_sqrt(float x) { return 1.0f / constexpr_sqrt(x); }

template <int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
PTO_INTERNAL void softmax_opt_fa_init_impl(
    TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x, ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
    ReduceTileD1 __out__ new_global_max, ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max,
    TileDataS1 __out__ tmp_float, TileDataS1 __out__ p_tile_f32)
{
    (void)local_max;
    (void)exp_max;
    (void)local_sum;

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    using Tile1D_fp32 = Tile<
        TileType::Vec, float, 1, TileDataS1::Rows * TileDataS1::Cols, BLayout::RowMajor, 1,
        TileDataS1::Rows * TileDataS1::Cols>;
    using Tile1D_out = Tile<
        TileType::Vec, typename TileDataD2::DType, 1, TileDataS1::Rows * TileDataS1::Cols, BLayout::RowMajor, 1,
        TileDataS1::Rows * TileDataS1::Cols>;
    Tile1D_fp32 p_tile_f32_1d;
    Tile1D_out x_exp_1d;

    // FA2.0 init mode
    TROWMAX(new_global_max, input_x, tmp_float);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB_IMPL(tmp_float, input_x, new_global_max);
    TMULS(tmp_float, tmp_float, scale);
    TEXP(p_tile_f32, tmp_float);
    pipe_barrier(PIPE_V);
    TROWSUM(new_global_sum, p_tile_f32, tmp_float);
    // TCVT(x_exp, p_tile_f32, RoundMode::CAST_ROUND);
    TRESHAPE(p_tile_f32_1d, p_tile_f32);
    TRESHAPE(x_exp_1d, x_exp);
    TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
}

template <int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
PTO_INTERNAL void softmax_opt_fa_not_init_impl(
    TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x, ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
    ReduceTileD1 __out__ new_global_max, ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max,
    TileDataS1 __out__ tmp_float, TileDataS1 __out__ p_tile_f32)
{
    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);

    using ReduceTileD2 = Tile<TileType::Vec, float, 1, ReduceTileD1::Rows, BLayout::RowMajor, 1, ReduceTileD1::Rows>;
    using Tile1D_fp32 = Tile<
        TileType::Vec, float, 1, TileDataS1::Rows * TileDataS1::Cols, BLayout::RowMajor, 1,
        TileDataS1::Rows * TileDataS1::Cols>;
    using Tile1D_out = Tile<
        TileType::Vec, typename TileDataD2::DType, 1, TileDataS1::Rows * TileDataS1::Cols, BLayout::RowMajor, 1,
        TileDataS1::Rows * TileDataS1::Cols>;

    ReduceTileD2 tmp_shw_local_max;
    ReduceTileD2 tmp_shw_new_global_max;
    ReduceTileD2 tmp_shw_exp_max;
    ReduceTileD2 tmp_shw_new_global_sum;
    ReduceTileD2 tmp_shw_local_sum;
    Tile1D_fp32 p_tile_f32_1d;
    Tile1D_out x_exp_1d;

    // FA2.0 not init mode
    TROWMAX(local_max, input_x, tmp_float);
    pipe_barrier(PIPE_V);
    TRESHAPE(tmp_shw_local_max, local_max);
    TRESHAPE(tmp_shw_new_global_max, new_global_max);
    TMAX(tmp_shw_local_max, tmp_shw_local_max, tmp_shw_new_global_max);
    pipe_barrier(PIPE_V);
    TRESHAPE(tmp_shw_exp_max, exp_max);
    TSUB(tmp_shw_exp_max, tmp_shw_new_global_max, tmp_shw_local_max);
    pipe_barrier(PIPE_V);

    TMULS(new_global_max, local_max, 1.0f); // just copy
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB_IMPL(tmp_float, input_x, local_max);
    TMULS(tmp_shw_exp_max, tmp_shw_exp_max, scale);
    TMULS(tmp_float, tmp_float, scale);
    TEXP(tmp_shw_exp_max, tmp_shw_exp_max);
    TRESHAPE(tmp_shw_exp_max, exp_max);
    TEXP(p_tile_f32, tmp_float);
    TRESHAPE(tmp_shw_exp_max, exp_max);
    // TCVT(x_exp, p_tile_f32, RoundMode::CAST_ROUND);
    TRESHAPE(p_tile_f32_1d, p_tile_f32);
    TRESHAPE(x_exp_1d, x_exp);
    TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
    pipe_barrier(PIPE_V);
    TRESHAPE(tmp_shw_new_global_sum, new_global_sum);
    TMUL(tmp_shw_new_global_sum, tmp_shw_exp_max, tmp_shw_new_global_sum);
    TROWSUM(local_sum, p_tile_f32, tmp_float);
    TRESHAPE(tmp_shw_local_sum, local_sum);
    pipe_barrier(PIPE_V);
    TADD(tmp_shw_new_global_sum, tmp_shw_new_global_sum, tmp_shw_local_sum);
}

template <bool init = false, int HEAD_SIZE, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
PTO_INTERNAL void pto_macro_fa_softmax(
    TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x, ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
    ReduceTileD1 __in__ new_global_max, ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max,
    TileDataS1 __out__ input_reduce_tmp, TileDataS1 __out__ p_tile_fp32)
{
    if constexpr (init) {
        softmax_opt_fa_init_impl<HEAD_SIZE, ReduceTileD1, TileDataD2, TileDataS1>(
            x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp,
            p_tile_fp32);
    } else {
        softmax_opt_fa_not_init_impl<HEAD_SIZE, ReduceTileD1, TileDataD2, TileDataS1>(
            x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp,
            p_tile_fp32);
    }
}

} // namespace pto

#endif
