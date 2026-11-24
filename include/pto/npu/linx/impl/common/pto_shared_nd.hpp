/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_LINX_NATIVE_PTO_SHARED_ND_HPP
#define PTO_LINX_NATIVE_PTO_SHARED_ND_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include <cstddef>

namespace pto::range {

// Source-only carrier for PTO-TMOV-SHARED-ND-TO-CUBE-001 (pto-spec PR #423,
// 6160ebf740eee788d9be4e2fb86fba398445241d). It deliberately does not model
// an ordinary Tile or widen the CUBE-only range::Subview operation domain.
template <typename Parent, std::size_t LengthUnits, unsigned OffsetUnits = 0,
          bool RuntimeBase = false>
class SharedNDSubview {
  static_assert(is_shared_tile_v<Parent> && !is_shared_tile_slot_v<Parent>,
                "Shared ND subview requires a compiler-managed SharedTile");
  static_assert(Parent::isRowMajor && !Parent::isBoxedLayout &&
                    !Parent::IsCubeLayout,
                "Shared ND subview requires unboxed RowMajor storage");
  static_assert(LengthUnits > 0 && LengthUnits <= 2048 &&
                    subview_size_code_for_bytes(LengthUnits * 128) != 0,
                "Shared ND subview length must be 128B..256KiB power of two");
  static_assert(is_valid_uimm11(OffsetUnits),
                "Shared ND subview offset must be 0..2047 units of 128B");
  static_assert(OffsetUnits * 128 + LengthUnits * 128 <= Parent::LogicalTileBytes,
                "Shared ND subview range must fit the parent capacity");

public:
  using ParentTile = Parent;
  static constexpr std::size_t CapacityBytes = LengthUnits * 128;
  static constexpr unsigned SubviewSizeCode =
      subview_size_code_for_bytes(CapacityBytes);
  static constexpr unsigned Offset = OffsetUnits;
  static constexpr bool HasRuntimeBase = RuntimeBase;

  explicit SharedNDSubview(const Parent &parent)
    requires(!RuntimeBase)
      : parent_(parent), base_units_(0) {}
  SharedNDSubview(const Parent &parent, uintptr_t base_units)
    requires(RuntimeBase)
      : parent_(parent), base_units_(base_units) {}
  SharedNDSubview(const Parent &&) = delete;
  SharedNDSubview(const Parent &&, uintptr_t) = delete;
  const Parent &parent() const { return parent_; }
  uintptr_t GetRangeBase() const { return base_units_; }

private:
  const Parent &parent_;
  uintptr_t base_units_;
};

template <std::size_t LengthUnits, unsigned OffsetUnits = 0, typename Parent>
auto shared_nd_subview(const Parent &parent) {
  return SharedNDSubview<Parent, LengthUnits, OffsetUnits>{parent};
}

template <std::size_t LengthUnits, unsigned OffsetUnits = 0, typename Parent>
auto shared_nd_subview(const Parent &parent, uintptr_t base_units) {
  return SharedNDSubview<Parent, LengthUnits, OffsetUnits, true>{parent, base_units};
}

template <std::size_t LengthUnits, unsigned OffsetUnits = 0, typename Parent>
auto shared_nd_subview(const Parent &&) = delete;
template <std::size_t LengthUnits, unsigned OffsetUnits = 0, typename Parent>
auto shared_nd_subview(const Parent &&, uintptr_t) = delete;

} // namespace pto::range

namespace pto::detail {

// Geometry validation is specific to this conversion, rather than generic
// B.SUBVIEW legality. A fragment keeps every physical column of its parent.
template <typename Cube, typename SharedND, bool Subview,
          std::size_t FragmentBytes, unsigned OffsetUnits = 0,
          bool RuntimeBase = false>
struct SharedNDToCubeContract {
  using ND = typename SharedND::LocalTileType;
  static_assert(Cube::BFractal == BLayout::CubeM32 ||
                    Cube::BFractal == BLayout::CubeM16,
                "Shared-ND-to-CUBE TMOV requires Local CUBE_M32 or CUBE_M16");
  static_assert(!Cube::isBoxedLayout,
                "Shared-ND-to-CUBE TMOV requires unboxed Local storage");
  static_assert(ND::isRowMajor && !ND::isBoxedLayout && !ND::IsCubeLayout,
                "Shared-ND-to-CUBE TMOV source must be RowMajor Shared ND");
  static_assert(std::is_same_v<typename ND::DType, typename Cube::DType>,
                "Shared-ND-to-CUBE TMOV preserves the backing dtype");
  static_assert(type_traits<typename ND::DType>::TypeCode != __type_rcpe6m2,
                "Shared-ND-to-CUBE TMOV rejects RCPE6M2 backing");
  static_assert(Cube::IsValidActiveSize && ND::IsValidActiveSize,
                "Shared-ND-to-CUBE TMOV requires assigned 128B..256KiB sizes");
  static_assert(ND::Rows * ND::Cols * tile_element_bits_v<typename ND::DType> ==
                    ND::LogicalTileBytes * 8,
                "Shared-ND-to-CUBE TMOV parent geometry must exactly cover capacity");
  static_assert(Cube::PhysicalCols == ND::PhysicalCols,
                "Shared-ND-to-CUBE TMOV requires matching physical columns");
  static_assert(FragmentBytes == Cube::LogicalTileBytes,
                "Shared-ND-to-CUBE TMOV fragment must equal destination capacity");
  static constexpr std::size_t RowBytes =
      (ND::PhysicalCols * tile_element_bits_v<typename ND::DType> + 7) / 8;
  static constexpr int Rows = Cube::PhysicalRows;
  static_assert(FragmentBytes == Rows * RowBytes,
                "Shared-ND-to-CUBE TMOV fragment must contain 32/16 complete rows");
  static_assert(Subview || (ND::PhysicalRows == Rows &&
                            ND::LogicalTileBytes == FragmentBytes),
                "Shared-ND-to-CUBE TMOV whole parent must match destination geometry");
  static_assert(RuntimeBase || (OffsetUnits * 128) % RowBytes == 0,
                "Shared-ND-to-CUBE TMOV subview must start at a complete row");

  struct Shape {
    bool valid;
    int rows;
    int columns;
  };

  // Called only by participating PEs. Check the sum before multiplication so
  // a wrapping GPR base can never masquerade as a valid range.
  static constexpr Shape shape(uintptr_t base_units, int parent_valid_rows,
                               int parent_valid_columns) {
    if (base_units > 2047 - OffsetUnits || parent_valid_rows <= 0 ||
        parent_valid_rows > ND::PhysicalRows || parent_valid_columns <= 0 ||
        parent_valid_columns > ND::PhysicalCols)
      return {false, 0, 0};
    const std::size_t offset_bytes = (base_units + OffsetUnits) * 128;
    if (offset_bytes % RowBytes != 0 ||
        offset_bytes + FragmentBytes > ND::LogicalTileBytes)
      return {false, 0, 0};
    const int origin_row = offset_bytes / RowBytes;
    const int remaining = parent_valid_rows - origin_row;
    const int valid_rows = remaining < Rows ? remaining : Rows;
    return {valid_rows > 0, valid_rows, parent_valid_columns};
  }
};

} // namespace pto::detail

#endif
