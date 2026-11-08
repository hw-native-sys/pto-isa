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

#ifndef PTO_LINX_VENDOR_LAYOUT_HPP
#define PTO_LINX_VENDOR_LAYOUT_HPP

#include <stdint.h>
#include <stddef.h>

#ifndef __linx
#include <iostream>
#include <stdio.h>
#endif
#include <type_traits>

#include "pto/npu/linx/impl/common/math_utils.hpp"

namespace pto {
enum class Location {
    Vec,
    Mat,
    Left,
    Right,
    Acc,
    Bias,
    Scaling,
    // v5: storage-class marker for compiler-managed Shared tiles. SharedTile
    // wraps a Local Tile and only changes its storage class (Local -> Shared),
    // never its matrix role / shape / dtype / layout. Used by TMATMUL Shared
    // matrix lowering (B.IOS binder) — see SharedTile<LocalTile>.
    Shared,
};

// PTOAS historically spells the tile role enum as TileType. Keep the Linx
// target source-compatible while retaining Location as the canonical API name.
using TileType = Location;

enum class BLayout {
    RowMajor,
    ColMajor,
    CubeM16,
    CubeM32,
    CubeN8,
};

enum class SLayout {
    NoneBox,
    RowMajor,
    ColMajor,
};

// Linx v0.58 does not encode Ascend compact modes. The trailing template
// parameter is accepted so PTOAS can share one typed EmitC surface across
// targets; non-Null values remain a compile-time descriptor only.
enum class CompactMode {
    Null,
    Normal,
    RowPlusOne,
};

enum class LayoutEnum {
    kNoneBox = 0,
    kRowMajor = 1,
    kColMajor = 2,
};

enum LayoutCvtEnum : uint64_t {
    NORM = 0,
    ND2DN = 1,
    ND2ZN = 2,
    ND2NZ = 3,
    DN2ND = 4,
    DN2ZN = 5,
    DN2NZ = 6,
    ZN2ND = 7,
    ZN2DN = 8,
    ZN2NZ = 9,
    NZ2ND = 10,
    NZ2DN = 11,
    NZ2ZN = 12,
    ND2M32 = 21,
    ND2M16 = 22,
    ND2N8 = 23,
    M322ND = 24,
    M162ND = 25,
    N82ND = 26,
    // PTO-ISA #291 direct-Local layout selectors (B.DATR Layout 29/31). Unlike
    // the ND2M32..N82ND transport conversions above, these name the operand's
    // physical Local layout itself instead of a GM<->Local transfer, so they are
    // legal on every operation that selects a Local operand layout.
    CUBE_M32 = 29,
    CUBE_M16 = 31,
};

// PTO weight-only TLOAD layouts.  These are B.DATR Layout codes, not ordinary
// Local/Shared storage layouts.  The selected source is converted to the
// existing row-major Shared [N][K] view; codes 12 and 13 (the future KN forms)
// are deliberately not exposed until PTO assigns them.
enum WeightLayoutEnum : uint64_t {
    OHWI2NK = 10,
    OIHW2NK = 11,
};

enum PadValueEnum : uint64_t { Zero = 0, Max = 1, Min = 2, Null = 3 };

const char* layout_type_to_str(LayoutEnum type)
{
    switch (type) {
        case LayoutEnum::kRowMajor:
            return "RowMajor";
        case LayoutEnum::kColMajor:
            return "ColMajor";
        case LayoutEnum::kNoneBox:
            return "kNoneBox";
    }
    return "UnsupportedLayout";
}

#ifndef __linx
class MatrixLayoutPrettyPrinter {
    template <typename Layout>
    static void print(std::ostream& out, const Layout& layout)
    {
        out << layout_type_to_str(Layout::kType) << "<" << Layout::Rows << ", " << Layout::Cols << ">, Strides<"
            << Layout::RowStride << ", " << Layout::ColStride << ">, Numel = " << Layout::Numel;
    }
};
#endif

template <
    const int Rows_, const int Cols_, const int RowStride_, const int ColStride_,
    const LayoutEnum kType_ = ColStride_ == 1 ? LayoutEnum::kRowMajor : LayoutEnum::kColMajor>
struct MatrixLayout {
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;

    static constexpr int RowStride = RowStride_;
    static constexpr int ColStride = ColStride_;

    static constexpr int Numel = Rows * Cols;

    static constexpr LayoutEnum kType = kType_;

    int operator()(int i, int j) const { return i * RowStride + j * ColStride; }
};

template <const int Row, const int Col, const int kStride = Col>
using RowMajor = MatrixLayout<Row, Col, kStride, 1>;

template <const int Row, const int Col, const int kStride = Row>
using ColMajor = MatrixLayout<Row, Col, 1, kStride>;

template <typename Layout>
static constexpr size_t num_rows = Layout::Rows;

template <typename Layout>
static constexpr size_t num_cols = Layout::Cols;

template <typename Layout>
static constexpr size_t row_stride = Layout::RowStride;

template <typename Layout>
static constexpr size_t col_stride = Layout::ColStride;

template <typename Layout>
static constexpr size_t get_numel = Layout::Numel;

template <typename Layout_>
static constexpr LayoutEnum layout_type = Layout_::kType;

template <typename Layout_>
struct is_row_major {
    static constexpr bool value = Layout_::kType == LayoutEnum::kRowMajor;
};

template <typename Layout_>
struct is_col_major {
    static constexpr bool value = Layout_::kType == LayoutEnum::kColMajor;
};

template <typename Layout>
struct is_contiguous {
    static constexpr bool value = is_row_major<Layout>::value ?
                                      (Layout::RowStride == Layout::Cols && Layout::ColStride == 1) :
                                      (Layout::ColStride == Layout::Rows && Layout::RowStride == 1);
};

template <typename OuterLayout_, typename InnerLayout_>
struct BlockMatrixLayout {
    using InnerLayout = InnerLayout_;
    using OuterLayout = OuterLayout_;

    static constexpr int Rows = OuterLayout_::Rows;
    static constexpr int Cols = OuterLayout_::Cols;
    static constexpr int Numel = OuterLayout_::Numel;

    static constexpr int kInnerRows = InnerLayout_::Rows;
    static constexpr int kInnerCols = InnerLayout_::Cols;

    static_assert(Rows % kInnerRows == 0, "OuterLayout rows must be divisible by InnerLayout rows");
    static_assert(Cols % kInnerCols == 0, "OuterLayout cols must be divisible by InnerLayout cols");

    static constexpr int kInnerNumel = InnerLayout_::Numel;
    static constexpr LayoutEnum kType = OuterLayout::kType;

    static constexpr int kTileRows = Rows / kInnerRows;
    static constexpr int kTileCols = Cols / kInnerCols;

    static constexpr bool kIsRowMajor = is_row_major<OuterLayout>::value;
    static constexpr bool kIsContiguous = is_contiguous<OuterLayout>::value;

    static constexpr int RowStride =
        kIsContiguous ? (kIsRowMajor ? kTileCols * kInnerNumel : kInnerNumel) : OuterLayout::RowStride;
    static constexpr int ColStride =
        kIsContiguous ? (kIsRowMajor ? kInnerNumel : kTileRows * kInnerNumel) : OuterLayout::ColStride;

    int operator()(int i, int j) const
    {
        const int outer_i = RowDivMod::div(i);
        const int outer_j = ColDivMod::div(j);

        const int inner_i = RowDivMod::mod(i);
        const int inner_j = ColDivMod::mod(j);

        return outer_(outer_i, outer_j) + inner_(inner_i, inner_j);
    }

    void dump() const
    {
#ifndef __linx
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                printf("%d, ", operator()(i, j));
            }
            printf("\n");
        }
#endif
    }

    auto get_outer_layout() const { return decltype(outer_){}; }

private:
    static constexpr bool kInnerRowsIsPow2 = (kInnerRows & (kInnerRows - 1)) == 0;
    static constexpr bool kInnerColsIsPow2 = (kInnerCols & (kInnerCols - 1)) == 0;

    using RowDivMod = DivModSelector<kInnerRowsIsPow2, kInnerRows>;
    using ColDivMod = DivModSelector<kInnerColsIsPow2, kInnerCols>;

    using BlockOuter = MatrixLayout<kTileRows, kTileCols, RowStride, ColStride, OuterLayout::kType>;
    BlockOuter outer_;
    InnerLayout inner_;
};

/// @brief Pretty printer for BlockMatrixLayout
#ifndef __linx
template <typename OuterLayout_, typename InnerLayout_>
static std::ostream& operator<<(std::ostream& out, const BlockMatrixLayout<OuterLayout_, InnerLayout_>& layout)
{
    out << "BlockMatrixLayout {" << std::endl
        << "    Outer: " << layout.get_outer_layout() << ", " << std::endl
        << "    Inner: " << InnerLayout_{} << std::endl
        << "  }";
    return out;
}
#endif

template <typename OuterLayout_, typename InnerLayout_>
concept BlockRowMajorLayout = is_row_major<OuterLayout_>::value && is_row_major<InnerLayout_>::value;

template <typename OuterLayout_, typename InnerLayout_>
concept BlockColMajorLayout = is_col_major<OuterLayout_>::value && is_col_major<InnerLayout_>::value;

template <typename OuterLayout_, typename InnerLayout_>
concept BlockMixedLayout = (is_row_major<OuterLayout_>::value && is_col_major<InnerLayout_>::value) ||
                           (is_col_major<OuterLayout_>::value && is_row_major<InnerLayout_>::value);

template <typename OuterLayout_, typename InnerLayout_>
    requires BlockRowMajorLayout<OuterLayout_, InnerLayout_>
using BlockRowMajor = BlockMatrixLayout<OuterLayout_, InnerLayout_>;

template <typename OuterLayout_, typename InnerLayout_>
    requires BlockColMajorLayout<OuterLayout_, InnerLayout_>
using BlockColMajor = BlockMatrixLayout<OuterLayout_, InnerLayout_>;

template <typename OuterLayout_, typename InnerLayout_>
    requires BlockMixedLayout<OuterLayout_, InnerLayout_>
using BlockMixed = BlockMatrixLayout<OuterLayout_, InnerLayout_>;
} // namespace pto

#endif
