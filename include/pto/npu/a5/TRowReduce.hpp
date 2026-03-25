/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

/**
 * @file TRowReduce.hpp
 * @brief 行归约操作实现（ROWSUM/ROWMAX/ROWMIN）
 *
 * 本文件实现了针对矩阵按行进行归约操作的算子，支持以下三种操作：
 * - TRowSum: 按行求和
 * - TRowMax: 按行求最大值
 * - TRowMin: 按行求最小值
 *
 * 支持的数据类型：half, float, int32_t, int16_t
 *
 * @note A5架构特殊说明：
 *   - vcadd 对 int16 输入产生 int32 输出（需要类型转换）
 *   - vcmax/vcmin 对 int16 输入输出均为 int16（无需类型转换）
 *   - int16 ROWSUM: int32中间结果转int16时采用回绕溢出（wrap-around），
 *     即截断高16位，剩余16位解释为有符号int16，与numpy行为一致
 */

#ifndef __ROW_REDUCE__
#define __ROW_REDUCE__

#include "common.hpp"
#include "pto/common/pto_tile.hpp"
#include "TPartBinOps.hpp"
#include <math.h>
#include <type_traits>

namespace pto {

//=============================================================================
// 归约操作策略（Policy Pattern）
//=============================================================================

/**
 * @brief 通用ROWSUM策略模板
 * @tparam T 数据类型（float, half, int32_t）
 *
 * 对于 float/half/int32，vcadd指令的输入输出类型相同。
 */
template <typename T>
struct ROWSUM {
    static constexpr typename Padding<T>::Type InitVal = Padding<T>::Zero;
    using TIN = T;  ///< 输入类型
    using TOUT = T; ///< 输出类型（与输入相同）

    /**
     * @brief 累加操作：dst = src0 + src1
     */
    static PTO_INTERNAL void Accumulate(RegTensor<TOUT> &dst, RegTensor<TOUT> &src0, RegTensor<TOUT> &src1,
                                        MaskReg &pred)
    {
        vadd(dst, src0, src1, pred, MODE_ZEROING);
    }

    /**
     * @brief 归约操作：对向量元素求和
     * @note vcadd将向量内所有元素相加，输出单个标量值
     */
    static PTO_INTERNAL void Reduce(RegTensor<TOUT> &dst, RegTensor<TIN> &src, MaskReg &pred)
    {
        vcadd(dst, src, pred, MODE_ZEROING);
    }
};

/**
 * @brief ROWSUM的int16特化版本
 *
 * @note A5架构关键特性：vcadd对int16输入产生int32输出
 *   - 输入：int16向量
 *   - 输出：int32标量（防止累加溢出）
 *   - 最终结果需通过vcvt转换回int16
 */
template <>
struct ROWSUM<int16_t> {
    static constexpr typename Padding<int32_t>::Type InitVal = Padding<int32_t>::Zero;
    using TIN = int16_t;  ///< 输入类型
    using TOUT = int32_t; ///< 中间计算类型（int32防止溢出）

    static PTO_INTERNAL void Accumulate(RegTensor<TOUT> &dst, RegTensor<TOUT> &src0, RegTensor<TOUT> &src1,
                                        MaskReg &pred)
    {
        vadd(dst, src0, src1, pred, MODE_ZEROING);
    }

    static PTO_INTERNAL void Reduce(RegTensor<TOUT> &dst, RegTensor<TIN> &src, MaskReg &pred)
    {
        vcadd(dst, src, pred, MODE_ZEROING);
    }
};

/**
 * @brief ROWMAX策略：按行求最大值
 * @tparam T 数据类型
 */
template <typename T>
struct ROWMAX {
    static constexpr typename Padding<T>::Type InitVal = Padding<T>::Min; ///< 初始值为最小值
    using TIN = T;
    using TOUT = T;

    static PTO_INTERNAL void Accumulate(RegTensor<TOUT> &dst, RegTensor<TOUT> &src0, RegTensor<TOUT> &src1,
                                        MaskReg &pred)
    {
        vmax(dst, src0, src1, pred, MODE_ZEROING);
    }

    static PTO_INTERNAL void Reduce(RegTensor<TOUT> &dst, RegTensor<TIN> &src, MaskReg &pred)
    {
        vcmax(dst, src, pred, MODE_ZEROING);
    }
};

/**
 * @brief ROWMIN策略：按行求最小值
 * @tparam T 数据类型
 */
template <typename T>
struct ROWMIN {
    static constexpr typename Padding<T>::Type InitVal = Padding<T>::Max; ///< 初始值为最大值
    using TIN = T;
    using TOUT = T;

    static PTO_INTERNAL void Accumulate(RegTensor<TOUT> &dst, RegTensor<TOUT> &src0, RegTensor<TOUT> &src1,
                                        MaskReg &pred)
    {
        vmin(dst, src0, src1, pred, MODE_ZEROING);
    }

    static PTO_INTERNAL void Reduce(RegTensor<TOUT> &dst, RegTensor<TIN> &src, MaskReg &pred)
    {
        vcmin(dst, src, pred, MODE_ZEROING);
    }
};

//=============================================================================
// 参数校验
//=============================================================================

/**
 * @brief 编译期和运行期参数校验
 * @tparam TileDataOut 输出Tile类型
 * @tparam TileDataIn 输入Tile类型
 * @tparam idx 是否是输出idx场景
 */
template <typename TileDataOut, typename TileDataIn, bool idx = false>
PTO_INTERNAL void TRowReduceCheck(uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValidRow)
{
    using T = typename TileDataIn::DType;
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>,
        "Row reduction only supports 'half', 'float', 'int32', or 'int16' data types. "
        "Fix: Define TileDataIn with DType = half, float, int32, or int16.");
    static_assert(idx || std::is_same_v<T, typename TileDataOut::DType>,
                  "Input and output tile data types must match. "
                  "Fix: Ensure TileDataOut uses the same DType as TileDataIn.");
    static_assert(!idx || std::is_same_v<uint32_t, typename TileDataOut::DType>,
                  "Output tile data type must be uint32_t. ");
    static_assert(TileDataOut::Loc == pto::TileType::Vec && TileDataIn::Loc == pto::TileType::Vec,
                  "Row reduction only works on vector tiles (TileType::Vec). "
                  "Fix: Instantiate TileDataIn and TileDataOut with Loc_ = TileType::Vec.");
    static_assert(TileDataIn::isRowMajor && !TileDataIn::isBoxedLayout,
                  "Input tile must use standard ND layout (row-major, non-fractal). "
                  "Fix: Define TileDataIn with BFractal_ = BLayout::RowMajor and SFractal_ "
                  "= SLayout::NoneBox, e.g.,\n"
                  "     Tile<TileType::Vec, T, ROWS, COLS, BLayout::RowMajor, ..., "
                  "SLayout::NoneBox>");
    static_assert((!TileDataOut::isBoxedLayout &&
                   (TileDataOut::isRowMajor || (!TileDataOut::isRowMajor && TileDataOut::Cols == 1))),
                  "Output tile layout must be either:\n"
                  "  (a) ND layout: BLayout::RowMajor + SLayout::NoneBox, OR\n"
                  "  (b) DN layout with exactly one column: BLayout::ColMajor + "
                  "SLayout::NoneBox + Cols=1.\n"
                  "Fix: Choose one of the following for TileDataOut:\n"
                  "     - Tile<..., ROWS, COLS, BLayout::RowMajor, ValidRows, 1>   // ND\n"
                  "     - Tile<..., ROWS, 1, BLayout::ColMajor, ValidRows, 1>  // DN with Cols=1");
    // runtime checks
    PTO_ASSERT(srcValidRows != 0 && srcValidCols != 0,
               "Source valid rows or columns is zero — row reduction requires at "
               "least one element per row. "
               "Fix: Ensure srcValidRows > 0 and srcValidCols > 0.");
    PTO_ASSERT(srcValidRows == dstValidRow,
               "Input and output valid row counts must be equal in row reduction "
               "(row count is preserved). "
               "Fix: Pass dstValidRow = srcValidRows.");
}

//=============================================================================
// 核心实现
//=============================================================================

/**
 * @brief 行归约核心实现
 *
 * 算法流程（每行）：
 *   1. 初始化累加器为初始值（SUM→0, MAX→MIN, MIN→MAX）
 *   2. 按elementsPerRepeat分块处理每行数据
 *   3. 对每块执行Reduce（vcadd/vcmax/vcmin）
 *   4. 将Reduce结果累加到累加器
 *   5. 如需类型转换（TOUT != TDST），执行vcvt后存储
 *
 * @tparam ReduceOp 归约策略（ROWSUM/ROWMAX/ROWMIN）
 * @tparam TileDataOut 输出Tile类型
 * @tparam TileDataIn 输入Tile类型
 * @tparam elementsPerRepeat 每次迭代处理的元素数
 *
 * @param dstPtr 输出缓冲区指针
 * @param srcPtr 输入缓冲区指针
 * @param rows 行数
 * @param cols 列数
 * @param version 实现版本（默认/无POST_UPDATE）
 */
template <typename ReduceOp, typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
PTO_INTERNAL void TRowReduceImpl(__ubuf__ typename TileDataOut::DType *dstPtr,
                                 __ubuf__ typename TileDataOut::DType *srcPtr, uint32_t rows, uint32_t cols,
                                 unsigned version)
{
    using TIN = typename ReduceOp::TIN;       ///< 输入数据类型
    using TOUT = typename ReduceOp::TOUT;     ///< 归约中间结果类型
    using TDST = typename TileDataOut::DType; ///< 最终输出类型

    // 对于int32→int16转换，需要设置CTRL寄存器以启用非饱和（回绕溢出）模式
    // CTRL[60]=1, CTRL[59]=1: 非饱和模式（截断高16位）
    constexpr int SAT_MODE_BIT_60 = 60;
    constexpr int SAT_MODE_BIT_59 = 59;
    constexpr bool needsNonSatMode = std::is_same_v<TOUT, int32_t> && std::is_same_v<TDST, int16_t>;
    bool originalCtrl60 = false;
    bool originalCtrl59 = false;

    if constexpr (needsNonSatMode) {
        uint64_t originalCtrl = get_ctrl();
        originalCtrl60 = (originalCtrl & (1ULL << SAT_MODE_BIT_60)) != 0;
        originalCtrl59 = (originalCtrl & (1ULL << SAT_MODE_BIT_59)) != 0;
        set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_60));
        set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_59));
    }

    uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        // 寄存器分配
        RegTensor<TIN> vreg0;        ///< 加载输入数据
        RegTensor<TOUT> vreg1;       ///< Reduce结果
        RegTensor<TOUT> vregdst;     ///< 累加器
        RegTensor<TDST> vreg_result; ///< 最终结果（仅TOUT!=TDST时使用）

        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TDST, DistVST::DIST_ONEPT>())>();
        uint32_t destItems = 1;
        MaskReg pregdst = CreatePredicate<TIN>(destItems);

        if (version == VFIMPL_2D_NO_POST_UPDATE) {
            // 版本1：无POST_UPDATE，使用二维索引访问
            for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
                // Step 1: 初始化累加器
                vbr((RegTensor<typename Padding<TOUT>::Type> &)vregdst, ReduceOp::InitVal);
                uint32_t sreg = cols;

                // Step 2-4: 分块处理
                for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                    MaskReg preg = CreatePredicate<TIN>(sreg);
                    // 加载数据块
                    vlds(vreg0, srcPtr, i * TileDataIn::RowStride + j * elementsPerRepeat, NORM);
                    // 归约：向量→标量
                    ReduceOp::Reduce(vreg1, vreg0, preg);
                    // 累加到结果
                    ReduceOp::Accumulate(vregdst, vregdst, vreg1, pregdst);
                }

                // Step 5: 存储结果（必要时类型转换）
                if constexpr (!std::is_same_v<TOUT, TDST>) {
                    // int16 ROWSUM: int32 → int16 回绕溢出转换（截断高16位）
                    // CTRL寄存器已设置为非饱和模式
                    vcvt(vreg_result, vregdst, pregdst, RS_DISABLE, PART_EVEN);
                    vsts(vreg_result, dstPtr, i * TileDataOut::RowStride, distValue, pregdst);
                } else {
                    vsts(vregdst, dstPtr, i * TileDataOut::RowStride, distValue, pregdst);
                }
            }
        } else {
            // 版本2：使用POST_UPDATE优化地址计算
            for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
                vbr((RegTensor<typename Padding<TOUT>::Type> &)vregdst, ReduceOp::InitVal);
                __ubuf__ TIN *row_ptr = srcPtr + i * TileDataIn::RowStride;
                uint32_t sreg = cols;

                for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                    MaskReg preg = CreatePredicate<TIN>(sreg);
                    vlds(vreg0, row_ptr, elementsPerRepeat, NORM, POST_UPDATE);
                    ReduceOp::Reduce(vreg1, vreg0, preg);
                    ReduceOp::Accumulate(vregdst, vregdst, vreg1, pregdst);
                }

                if constexpr (!std::is_same_v<TOUT, TDST>) {
                    // int16 ROWSUM: int32 → int16 回绕溢出转换（截断高16位）
                    // CTRL寄存器已设置为非饱和模式
                    vcvt(vreg_result, vregdst, pregdst, RS_DISABLE, PART_EVEN);
                    vsts(vreg_result, dstPtr, TileDataOut::RowStride, distValue, pregdst, POST_UPDATE);
                } else {
                    vsts(vregdst, dstPtr, TileDataOut::RowStride, distValue, pregdst, POST_UPDATE);
                }
            }
        }
    } // end VF

    // 恢复原始CTRL寄存器状态
    if constexpr (needsNonSatMode) {
        if (originalCtrl60) {
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_60));
        } else {
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_60));
        }
        if (originalCtrl59) {
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_59));
        } else {
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_59));
        }
    }
}

//=============================================================================
// 算子入口
//=============================================================================

/**
 * @brief 按行求最大值
 * @tparam TileDataOut 输出Tile类型
 * @tparam TileDataIn 输入Tile类型
 * @tparam elementsPerRepeat 每次迭代处理的元素数
 */
template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWMAX)
    OP_TYPE(reduce) void TRowMax(typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
                                 uint32_t dstValidRow, uint32_t srcValidRows, uint32_t srcValidCols,
                                 unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    TRowReduceCheck<TileDataOut, TileDataIn>(srcValidRows, srcValidCols, dstValidRow);

    using TIN = typename TileDataIn::DType;
    __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

    using rowReduceOp = ROWMAX<typename TileDataIn::DType>;
    TRowReduceImpl<rowReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(dstPtr, srcPtr, srcValidRows, srcValidCols,
                                                                            version);
}

/**
 * @brief 按行求和
 * @tparam TileDataOut 输出Tile类型
 * @tparam TileDataIn 输入Tile类型
 * @tparam elementsPerRepeat 每次迭代处理的元素数
 */
template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWSUM)
    OP_TYPE(reduce) void TRowSum(typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
                                 uint32_t dstValidRow, uint32_t srcValidRows, uint32_t srcValidCols,
                                 unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    TRowReduceCheck<TileDataOut, TileDataIn>(srcValidRows, srcValidCols, dstValidRow);

    using TIN = typename TileDataIn::DType;
    __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

    using rowReduceOp = ROWSUM<typename TileDataIn::DType>;
    TRowReduceImpl<rowReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(dstPtr, srcPtr, srcValidRows, srcValidCols,
                                                                            version);
}

/**
 * @brief 按行求最小值
 * @tparam TileDataOut 输出Tile类型
 * @tparam TileDataIn 输入Tile类型
 * @tparam elementsPerRepeat 每次迭代处理的元素数
 */
template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWMIN)
    OP_TYPE(reduce) void TRowMin(typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
                                 uint32_t dstValidRow, uint32_t srcValidRows, uint32_t srcValidCols,
                                 unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    TRowReduceCheck<TileDataOut, TileDataIn>(srcValidRows, srcValidCols, dstValidRow);

    using TIN = typename TileDataIn::DType;
    __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

    using rowReduceOp = ROWMIN<typename TileDataIn::DType>;
    TRowReduceImpl<rowReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(dstPtr, srcPtr, srcValidRows, srcValidCols,
                                                                            version);
}

//=============================================================================
// 便捷封装（自动计算elementsPerRepeat）
//=============================================================================

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    using T = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();

    TRowMax<TileDataOut, TileDataIn, elementsPerRepeat>(dst.data(), src.data(), dst.GetValidRow(), rows, cols);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWSUM_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    using T = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();

    TRowSum<TileDataOut, TileDataIn, elementsPerRepeat>(dst.data(), src.data(), dst.GetValidRow(), rows, cols);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    using T = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();

    TRowMin<TileDataOut, TileDataIn, elementsPerRepeat>(dst.data(), src.data(), dst.GetValidRow(), rows, cols);
}

} // namespace pto

#endif