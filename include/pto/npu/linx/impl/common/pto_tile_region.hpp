/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_LINX_NATIVE_PTO_TILE_REGION_HPP
#define PTO_LINX_NATIVE_PTO_TILE_REGION_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include "pto/npu/linx/impl/common/pto_local_cube_assembly.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace pto {

// Rounding-mode selector used by standalone region inline-assembly helpers.
// Keep this in the common region header because those helpers are intentionally
// usable without including the larger jcore/template_asm.hpp header.
enum LinxRMode {
  LINX_RNONE = 0,
  LINX_RNE = 1,
  LINX_RTZ = 2,
  LINX_RDN = 3,
  LINX_RUP = 4,
  LINX_RNA = 5,
  LINX_RTO = 6,
  LINX_RHB = 7,
};

// B.DATR.Layout codes for Local elementwise operands.
// 29 selects CUBE_M32, 31 selects CUBE_M16, and 0 keeps the NORM default.
template <typename Tile>
inline constexpr int local_layout_code_v =
    Tile::BFractal == BLayout::CubeM32 ? 29
    : Tile::BFractal == BLayout::CubeM16 ? 31
                                         : 0;

// PTO-ISA #291 (issue #160 item 6): a TEPL tile operation selects its Local
// operand layout through B.DATR.Layout, so a CUBE_M16/M32 operand keeps its
// physical CUBE representation instead of collapsing to the NORM (RowMajor)
// default.  This macro mirrors template_asm.hpp's PTO_ELEMENTWISE_LAYOUT_ASM
// but lives in the common header so the region inline-asm paths can emit the
// same B.DATR without depending on template_asm.hpp's include order.  RowMajor
// keeps its historical no-B.DATR encoding.
#define PTO_REGION_ELEMENTWISE_LAYOUT_ASM                                      \
  ".if %c[ElemLayout] == 29\nB.DATR CUBE_M32, Null\n"                          \
  ".elseif %c[ElemLayout] == 31\nB.DATR CUBE_M16, Null\n"                      \
  ".endif\n"

// TMOV permits only the Layout attribute; its pad union is reserved-zero.
// Zero here is an encoding requirement, not a request to fill copied padding.
#define PTO_TMOV_LAYOUT_ASM                                                  \
  ".if %c[ElemLayout] == 29\nB.DATR CUBE_M32, Zero\n"                       \
  ".elseif %c[ElemLayout] == 31\nB.DATR CUBE_M16, Zero\n"                   \
  ".endif\n"

namespace region {

template <typename...>
inline constexpr bool dependent_false_v = false;

constexpr int tile_size_code_for_bytes(std::size_t bytes) {
  std::size_t size = 128;
  int code = 1;
  while (size < bytes && code < 12) {
    size <<= 1;
    ++code;
  }
  return size == bytes ? code : -1;
}

template <typename Parent, typename SubTile, int Rows, int Cols>
struct partition_contract {
  static_assert(is_tile<Parent>::value &&
                    range::is_legal_subview_parent_v<Parent>,
                "TPARTVIEW parent must be an assigned Local Matrix Tile with "
                "a CUBE layout");
  static_assert(Rows > 0 && Cols > 0,
                "Tile partition extents must be positive");
  static_assert(std::is_same_v<typename Parent::DType, typename SubTile::DType>,
                "Tile partition requires matching element types");
  static_assert(Parent::Loc == SubTile::Loc,
                "Tile partition requires matching locations");
  static_assert(Parent::BFractal == SubTile::BFractal,
                "Tile partition requires matching layouts");
  static_assert(Parent::SFractal == SubTile::SFractal,
                "Tile partition requires matching storage layouts");
  static_assert(SubTile::LogicalTileBytes >= 128,
                "Tile partition fragments must contain at least one CELL");
  static_assert(Parent::Rows == Rows * SubTile::Rows &&
                    Parent::Cols == Cols * SubTile::Cols,
                "Tile partition must exactly cover the parent physical shape");
  static_assert(Parent::ValidRow == Rows * SubTile::ValidRow &&
                    Parent::ValidCol == Cols * SubTile::ValidCol,
                "Tile partition must exactly cover the parent valid shape");
  static_assert(Parent::LogicalTileBytes ==
                    Rows * Cols * SubTile::LogicalTileBytes,
                "Tile partition must cover the complete parent Tile");
};

template <typename Parent, typename SubTile>
class SubTileView {
  static_assert(range::is_legal_subview_parent_v<Parent>,
                "B.SUBVIEW parent must be an assigned Local CUBE or Shared RowMajor "
                "tile layout");
public:
  using ParentTile = Parent;
  using SubTileType = SubTile;
  using DType = typename SubTile::DType;
  using TileDType = typename Parent::TileDType;
  static constexpr Location Loc = Parent::Loc;
  static constexpr int Rows = SubTile::Rows;
  static constexpr int Cols = SubTile::Cols;
  // B.DIM's third operand is the physical source column extent.  Keep this
  // explicitly on the view rather than deriving it from Parent: a view is a
  // bounded source carrier and its geometry is the SubTile geometry.
  static constexpr int PhysicalRows = SubTile::PhysicalRows;
  static constexpr int PhysicalCols = SubTile::PhysicalCols;
  static constexpr int PhysicalCol = PhysicalCols;
  static constexpr int RowStride = SubTile::RowStride;
  static constexpr int ColStride = SubTile::ColStride;
  static constexpr int ValidRow = SubTile::ValidRow;
  static constexpr int ValidCol = SubTile::ValidCol;
  static constexpr BLayout BFractal = SubTile::BFractal;
  static constexpr int CubeCellCols = SubTile::CubeCellCols;
  static constexpr SLayout SFractal = SubTile::SFractal;
  static constexpr int SFractalSize = SubTile::SFractalSize;
  static constexpr bool isRowMajor = SubTile::isRowMajor;
  static constexpr bool isBoxedLayout = SubTile::isBoxedLayout;
  static constexpr bool isInnerRowMajor = SubTile::isInnerRowMajor;
  static constexpr bool isInnerColMajor = SubTile::isInnerColMajor;
  static constexpr int InnerRows = SubTile::InnerRows;
  static constexpr int InnerCols = SubTile::InnerCols;
  static constexpr int Numel = SubTile::Numel;
  static constexpr int LogicalTileBytes = SubTile::LogicalTileBytes;
  static constexpr int TilesizeCode = SubTile::TilesizeCode;
  static constexpr bool IsValidActiveSize = SubTile::IsValidActiveSize;

  SubTileView(Parent &parent, int row, int col, int partition_cols)
      : parent_(&parent), row_(row), col_(col),
        partition_cols_(partition_cols) {}

  Parent &parent() const { return *parent_; }
  int row() const { return row_; }
  int col() const { return col_; }
  std::uintptr_t GetRangeBase() const {
    static_assert(SubTile::LogicalTileBytes % range::RangeAddressUnitBytes == 0,
                  "Tile partition offsets must be representable in 128B units");
    return static_cast<std::uintptr_t>(
        (static_cast<std::size_t>(row_) * partition_cols_ + col_) *
        (SubTile::LogicalTileBytes / range::RangeAddressUnitBytes));
  }
  int GetValidRow() const { return SubTile::ValidRow; }
  int GetValidCol() const { return SubTile::ValidCol; }
  decltype(auto) data() { return parent_->data(); }

private:
  Parent *parent_;
  int row_;
  int col_;
  int partition_cols_;
};

template <typename Parent, typename SubTile>
class ReductionPrefixView {
  static_assert(Parent::IsCubeLayout,
                "reduction prefix view requires persistent CUBE storage");
  static_assert(Parent::ValidCol == 1,
                "reduction prefix view requires a one-column valid result");
  static_assert(SubTile::BFractal == Parent::BFractal,
                "reduction prefix view must keep the parent CUBE layout");
  static_assert(SubTile::Rows == Parent::Rows,
                "reduction prefix view must span the full CUBE row count");
  // The single-CELL selection is expressed as an exact 128-byte extent below
  // (SubTile::LogicalTileBytes == range::RangeAddressUnitBytes), not a fixed
  // physical column count: one 128-byte CUBE CELL is [32,1] for FP32, [32,2]
  // for BF16 and [32,4] for E8M0.  Pinning SubTile::Cols == 1 would reject the
  // wider dtypes' legal first CELL (issue #160 item 4).
  static_assert(SubTile::ValidRow == Parent::ValidRow &&
                    SubTile::ValidCol == 1,
                "reduction prefix view must preserve the reduction valid shape");
  static_assert(std::is_same_v<typename Parent::DType, typename SubTile::DType>,
                "reduction prefix view requires matching element types");
  static_assert(SubTile::LogicalTileBytes == range::RangeAddressUnitBytes,
                "reduction prefix view must select one 128-byte CELL");

public:
  using ParentTile = Parent;
  using SubTileType = SubTile;
  using DType = typename SubTile::DType;
  using TileDType = typename Parent::TileDType;
  static constexpr Location Loc = Parent::Loc;
  static constexpr int Rows = SubTile::Rows;
  static constexpr int Cols = SubTile::Cols;
  static constexpr int PhysicalRows = SubTile::PhysicalRows;
  static constexpr int PhysicalCols = SubTile::PhysicalCols;
  static constexpr int RowStride = SubTile::RowStride;
  static constexpr int ColStride = SubTile::ColStride;
  static constexpr int ValidRow = SubTile::ValidRow;
  static constexpr int ValidCol = SubTile::ValidCol;
  static constexpr BLayout BFractal = SubTile::BFractal;
  static constexpr SLayout SFractal = SubTile::SFractal;
  static constexpr int CubeCellCols = SubTile::CubeCellCols;
  static constexpr bool IsCubeLayout = SubTile::IsCubeLayout;
  static constexpr int LogicalTileBytes = SubTile::LogicalTileBytes;
  static constexpr int TilesizeCode = SubTile::TilesizeCode;
  static constexpr bool IsValidActiveSize = SubTile::IsValidActiveSize;

  explicit ReductionPrefixView(Parent &parent) : parent_(&parent) {}

  using reduction_prefix_parent = Parent;

  Parent &parent() const { return *parent_; }
  decltype(auto) data() { return parent_->data(); }
  decltype(auto) data() const { return parent_->data(); }
  int GetValidRow() const { return ValidRow; }
  int GetValidCol() const { return ValidCol; }
  std::uintptr_t GetRangeBase() const { return 0; }

private:
  Parent *parent_;
};
template <typename Parent, typename SubTile, int Rows, int Cols>
class BorrowedTileArray {
  using Contract = partition_contract<Parent, SubTile, Rows, Cols>;
  static_assert(sizeof(Contract) > 0);

public:
  using ParentTile = Parent;
  using SubTileType = SubTile;
  static constexpr int rank = 2;
  static constexpr int rows = Rows;
  static constexpr int cols = Cols;
  static constexpr int PartCount = Parent::Numel / SubTile::Numel;
  static constexpr bool is_logical32_partition =
      is_element_tile_v<Parent> && is_element_tile_v<SubTile> &&
      (Parent::Numel == 128 || Parent::Numel == 512 ||
       ((Parent::IsElement32Type || Parent::IsElement64Type) &&
        !std::is_same_v<typename Parent::DType, float> &&
        (Parent::Numel == 64 || Parent::Numel == 256 ||
         Parent::Numel == 1024 ||
         (Parent::IsElement32Type && Parent::Numel == 2048))) ||
       (std::is_same_v<typename Parent::DType, float> &&
        (Parent::Numel == 2048 || Parent::Numel == 16384))) &&
      SubTile::Numel == 32 && Rows == 1 && Cols == PartCount;
  static constexpr bool is_logical64_partition =
      is_element_tile_v<Parent> && is_element_tile_v<SubTile> &&
      (std::is_same_v<typename Parent::DType, __half> ||
       std::is_same_v<typename Parent::DType, uint16_t> ||
       std::is_same_v<typename Parent::DType, uint32_t> ||
       std::is_same_v<typename Parent::DType, int32_t>) &&
      (Parent::Numel == 64 || Parent::Numel == 128 ||
       Parent::Numel == 512 || Parent::Numel == 2048) &&
      SubTile::Numel == 64 && Rows == 1 && Cols == PartCount;
  static constexpr bool is_element_partition =
      is_logical32_partition || is_logical64_partition;
  static_assert(!is_element_partition ||
                    static_cast<uint64_t>(Parent::Numel) <= UINT32_MAX,
                "ElementTile logical indices require a B32 parent capacity");

  explicit BorrowedTileArray(Parent &parent) : parent_(&parent) {}

  BorrowedTileArray(Parent &parent, std::size_t valid_elements)
    requires(is_element_partition)
      : parent_(&parent), valid_elements_(valid_elements) {
    if (valid_elements > static_cast<std::size_t>(Parent::Numel))
      __builtin_trap();
    valid_elements_b32_ = static_cast<uint32_t>(valid_elements);
  }

  Parent &parent() const { return *parent_; }

  std::size_t size() const
    requires(is_element_partition)
  {
    return static_cast<std::size_t>(Rows * Cols);
  }

  SubTileView<Parent, SubTile> part(std::size_t flat_index) const
    requires(is_element_partition)
  {
    if (flat_index >= size())
      __builtin_trap();
    return SubTileView<Parent, SubTile>(
        *parent_, static_cast<int>(flat_index / Cols),
        static_cast<int>(flat_index % Cols), Cols);
  }

  std::size_t valid_size(std::size_t part_index) const
    requires(is_element_partition)
  {
    if (part_index >= size())
      __builtin_trap();
    if constexpr (is_logical64_partition) {
      const std::size_t parent_columns =
          static_cast<std::size_t>(Parent::Cols);
      const std::size_t full_rows = valid_elements_ / parent_columns;
      const std::size_t tail = valid_elements_ % parent_columns;
      const std::size_t part_column = 2 * part_index;
      const std::size_t tail_elements =
          tail <= part_column ? 0 : tail - part_column < 2
                                      ? tail - part_column
                                      : 2;
      return 2 * full_rows + tail_elements;
    } else {
      if (part_index >= valid_elements_)
        return 0;
      const std::size_t remaining = valid_elements_ - part_index;
      const std::size_t part_elements =
          (remaining + static_cast<std::size_t>(Cols) - 1) /
          static_cast<std::size_t>(Cols);
      return part_elements < static_cast<std::size_t>(SubTile::Numel)
                 ? part_elements
                 : static_cast<std::size_t>(SubTile::Numel);
    }
  }

  constexpr uint32_t logical_index(std::size_t part_index,
                                   uint32_t element_index) const
    requires(is_element_partition)
  {
    if (part_index >= static_cast<std::size_t>(PartCount) ||
        element_index >= static_cast<uint32_t>(SubTile::Numel))
      __builtin_trap();
    const uint32_t part_index_b32 = static_cast<uint32_t>(part_index);
    const uint32_t index = is_logical64_partition
        ? (element_index / 2) * static_cast<uint32_t>(Parent::Cols) +
              2 * part_index_b32 + element_index % 2
        : part_index_b32 +
              static_cast<uint32_t>(PartCount) * element_index;
    if (index >= valid_elements_b32_)
      __builtin_trap();
    return index;
  }

  template <typename T>
    requires(is_element_partition &&
             std::is_same_v<T, typename Parent::DType>)
  auto logical_region(T *block_base, std::size_t part_index) const
      -> global_tensor<T, RowMajor<Parent::Rows, Parent::Cols>> {
    if (part_index >= size())
      __builtin_trap();
    return global_tensor<T, RowMajor<Parent::Rows, Parent::Cols>>(
        block_base + (is_logical64_partition ? 2 * part_index : part_index));
  }

  uint64_t store_predicate(std::size_t count) const
    requires(is_logical64_partition)
  {
    if (count > static_cast<std::size_t>(SubTile::Numel))
      __builtin_trap();
    const std::size_t low_count = (count + 1) / 2;
    const std::size_t high_count = count / 2;
    const auto low_bits = [](std::size_t bits) -> uint64_t {
      return bits == 0 ? 0 : (uint64_t{1} << bits) - 1;
    };
    return low_bits(low_count) | (low_bits(high_count) << 32);
  }

  class Row {
  public:
    Row(Parent &parent, int row) : parent_(&parent), row_(row) {}

    SubTileView<Parent, SubTile> operator[](int col) const {
      return SubTileView<Parent, SubTile>(*parent_, row_, col, Cols);
    }

  private:
    Parent *parent_;
    int row_;
  };

  Row operator[](int row) const { return Row(*parent_, row); }

  Row row(int row) const {
    return Row(*parent_, row);
  }

private:
  Parent *parent_;
  std::size_t valid_elements_ = static_cast<std::size_t>(Parent::Numel);
  uint32_t valid_elements_b32_ = static_cast<uint32_t>(Parent::Numel);
};


template <typename SubTile>
class TileArrayOutputRef;

template <typename SubTile, int Rows, int Cols>
class TileArray {
public:
  static_assert(Rows > 0 && Cols > 0,
                "TileArray extents must be positive");
  using SubTileType = SubTile;
  static constexpr int rows = Rows;
  static constexpr int cols = Cols;
  static constexpr std::size_t slot_count =
      static_cast<std::size_t>(Rows) * static_cast<std::size_t>(Cols);
  // One output slot emits one writer in the Local generation. The owning
  // B.ASSEMBLE contract admits at most 16 writers, including INIT and LAST.
  static_assert(slot_count <= 16,
                "TileArray assembly supports at most 16 writers per generation");
  static constexpr std::size_t ParentBytes =
      SubTile::LogicalTileBytes * slot_count;
  static constexpr int ParentSizeCode =
      tile_size_code_for_bytes(ParentBytes);
  static_assert((slot_count - 1) *
                    (SubTile::LogicalTileBytes / range::RangeAddressUnitBytes) <=
                2047,
                "Tile assembly slot range exceeds the ISA uimm11 limit");
  static_assert(ParentSizeCode > 0,
                "Tile assembly parent capacity needs a PTO SizeCode");

#if defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)
  using ParentCarrier =
      linx_tile_carrier<ParentBytes,
                        typename SubTile::TileDType::ScalarType>;
  using ParentRegisterType = typename ParentCarrier::RegisterType;
#else
  using ParentCarrier =
      typename SubTile::DType[ParentBytes * 8 /
                              type_traits<typename SubTile::DType>::bits];
#endif

  TileArray() = default;
  TileArray(const TileArray &) = delete;
  TileArray &operator=(const TileArray &) = delete;
  TileArray(TileArray &&) = default;
  TileArray &operator=(TileArray &&) = default;

#if defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)
  ParentRegisterType &data() { return data_.Register; }
  ParentCarrier &carrier() { return data_; }
#else
  ParentCarrier &data() { return data_; }
  ParentCarrier &carrier() { return data_; }
#endif

  class Row {
  public:
    Row(TileArray &array, int row) : array_(&array), row_(row) {}

    TileArrayOutputRef<SubTile> operator[](int col) {
      return array_->makeOutputRef(row_, col);
    }

  private:
    TileArray *array_;
    int row_;
  };

  Row operator[](int row) { return Row(*this, row); }
  Row row(int row) { return Row(*this, row); }

private:
  TileArrayOutputRef<SubTile> makeOutputRef(int row, int col) {
    return TileArrayOutputRef<SubTile>(data_, row, col, Cols,
                                       Rows * Cols, ParentSizeCode);
  }
  ParentCarrier data_;
};

template <typename SubTile>
class TileArrayOutputRef {
public:
  using SubTileType = SubTile;

  template <typename, int, int>
  friend class TileArray;

  int row() const { return row_; }
  int col() const { return col_; }
  int ordinal() const { return row_ * array_cols_ + col_; }
  int slot_count() const { return slot_count_; }
  std::uintptr_t range_base_units() const {
    static_assert(SubTile::LogicalTileBytes % range::RangeAddressUnitBytes == 0,
                  "Tile assembly offsets must be representable in 128B units");
    return static_cast<std::uintptr_t>(
        ordinal() * (SubTile::LogicalTileBytes /
                     range::RangeAddressUnitBytes));
  }
  static constexpr int Rows = SubTile::Rows;
  static constexpr int Cols = SubTile::Cols;
  static constexpr int PhysicalRows = SubTile::PhysicalRows;
  static constexpr int PhysicalCols = SubTile::PhysicalCols;
  static constexpr int ValidRow = SubTile::ValidRow;
  static constexpr int ValidCol = SubTile::ValidCol;
  static constexpr int RowStride = SubTile::RowStride;
  static constexpr int ColStride = SubTile::ColStride;
  static constexpr Location Loc = SubTile::Loc;
  static constexpr BLayout BFractal = SubTile::BFractal;
  static constexpr SLayout SFractal = SubTile::SFractal;
  static constexpr bool isRowMajor = SubTile::isRowMajor;
  static constexpr bool isBoxedLayout = SubTile::isBoxedLayout;
  static constexpr bool isInnerRowMajor = SubTile::isInnerRowMajor;
  static constexpr bool isInnerColMajor = SubTile::isInnerColMajor;
  static constexpr int InnerRows = SubTile::InnerRows;
  static constexpr int InnerCols = SubTile::InnerCols;
  static constexpr int Numel = SubTile::Numel;
  static constexpr int LogicalTileBytes = SubTile::LogicalTileBytes;
  static constexpr int TilesizeCode = SubTile::TilesizeCode;
  static constexpr bool IsValidActiveSize = SubTile::IsValidActiveSize;
  using DType = typename SubTile::DType;
  using TileDType = typename SubTile::TileDType;

  template <int ParentSizeCode>
  auto &parent_data() {
    static_assert(ParentSizeCode >= 1 && ParentSizeCode <= 12);
    constexpr std::size_t ParentBytes =
        static_cast<std::size_t>(128) << (ParentSizeCode - 1);
#if defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)
    using Carrier =
        linx_tile_carrier<ParentBytes,
                          typename SubTile::TileDType::ScalarType>;
    return reinterpret_cast<Carrier *>(array_)->Register;
#else
    using Carrier = typename SubTile::DType[
        ParentBytes * 8 / type_traits<typename SubTile::DType>::bits];
    return *reinterpret_cast<Carrier *>(array_);
#endif
  }
  int GetValidRow() const { return ValidRow; }
  int GetValidCol() const { return ValidCol; }
  int parent_size_code() const { return parent_size_code_; }

private:
  template <typename, int, int>
  friend class TileArray;
  template <typename Carrier>
  TileArrayOutputRef(Carrier &array, int row, int col, int array_cols,
                     int slot_count, int parent_size_code)
      : array_(&array), row_(row), col_(col), array_cols_(array_cols),
        slot_count_(slot_count), parent_size_code_(parent_size_code) {}
  void *array_ = nullptr;
  int row_ = 0;
  int col_ = 0;
  int array_cols_ = 0;
  int slot_count_ = 0;
  int parent_size_code_ = 0;
};

} // namespace region

template <typename Parent, typename SubTile>
using SubTileView = region::SubTileView<Parent, SubTile>;

template <typename Parent, typename SubTile, int Rows, int Cols>
using BorrowedTileArray =
    region::BorrowedTileArray<Parent, SubTile, Rows, Cols>;

template <typename SubTile, int Rows, int Cols>
using TileArray = region::TileArray<SubTile, Rows, Cols>;

template <typename SubTile>
using TileArrayOutputRef = region::TileArrayOutputRef<SubTile>;

template <typename T>
struct is_subtile_view : std::false_type {};
template <typename Parent, typename SubTile>
struct is_subtile_view<region::SubTileView<Parent, SubTile>> : std::true_type {};
template <typename Parent, typename SubTile>
struct is_subtile_view<region::ReductionPrefixView<Parent, SubTile>>
    : std::true_type {};
template <typename T>
inline constexpr bool is_subtile_view_v = is_subtile_view<T>::value;

// A TPARTVIEW fragment is a tile-shaped source operand: it carries the
// fragment's static geometry and supplies the B.SUBVIEW offset itself.
template <typename Parent, typename SubTile>
requires(range::is_legal_subview_parent_v<Parent>)
struct is_tile<region::SubTileView<Parent, SubTile>> : std::true_type {
  static constexpr SLayout layout_enum = SubTile::SFractal;
};

template <typename Parent, typename SubTile>
struct is_tile<region::ReductionPrefixView<Parent, SubTile>> : std::true_type {
  static constexpr SLayout layout_enum = SubTile::SFractal;
};

// Range-modifier trait for the reduction-prefix source carrier: borrows the
// first CELL of a wide CUBE reduction destination as a zero-copy source
// (PTO #311).
template <typename T> struct is_reduction_prefix_view : std::false_type {};
template <typename Parent, typename SubTile>
struct is_reduction_prefix_view<region::ReductionPrefixView<Parent, SubTile>>
    : std::true_type {};
template <typename T>
concept is_reduction_prefix_view_v = is_reduction_prefix_view<T>::value;

template <typename T>
struct is_tile_array_output_ref : std::false_type {};
template <typename SubTile>
struct is_tile_array_output_ref<region::TileArrayOutputRef<SubTile>>
    : std::true_type {};
template <typename T>
inline constexpr bool is_tile_array_output_ref_v =
    is_tile_array_output_ref<T>::value;

// Full-storage ElementTile fragments admitted by raw hierarchical assembly.
// Logical32/logical64 leaves retain their existing profiles. Wider fragments
// are limited to the closed storage levels used by the public hierarchy;
// arbitrary same-sized element profiles are not interchangeable.
template <typename T>
inline constexpr bool is_element_assembly_fragment_v = [] {
  using TileType = std::remove_cvref_t<T>;
  if constexpr (!is_element_tile_v<TileType>) {
    return false;
  } else {
    return TileType::Numel == 32 || TileType::Numel == 64 ||
           ((TileType::IsElement64Type ||
             (TileType::IsElement32Type &&
              !std::is_same_v<typename TileType::DType, float>)) &&
            (TileType::Numel == 128 || TileType::Numel == 256 ||
             TileType::Numel == 512)) ||
           (TileType::Numel == 128 &&
            (std::is_same_v<typename TileType::DType, float> ||
             std::is_same_v<typename TileType::DType, __half>)) ||
           (TileType::Numel == 2048 &&
            std::is_same_v<typename TileType::DType, float>);
  }
}();

template <typename SubTile, typename Parent>
auto TREDUCEPREFIXVIEW(Parent &parent)
    -> region::ReductionPrefixView<Parent, SubTile> {
  return region::ReductionPrefixView<Parent, SubTile>(parent);
}
template <typename SubTile, int Rows, int Cols, typename Parent>
  requires(range::is_legal_subview_parent_v<Parent>)
auto TPARTVIEW(Parent &parent)
    -> region::BorrowedTileArray<Parent, SubTile, Rows, Cols> {
  return region::BorrowedTileArray<Parent, SubTile, Rows, Cols>(parent);
}

template <int Elements, typename Parent>
  requires(is_element_tile_v<Parent>)
auto TPARTVIEW(Parent &parent, std::size_t valid_elements) {
  constexpr bool IsDense16 =
      std::is_same_v<typename Parent::DType, __half> ||
      std::is_same_v<typename Parent::DType, uint16_t>;
  constexpr bool IsInteger32 =
      std::is_same_v<typename Parent::DType, uint32_t> ||
      std::is_same_v<typename Parent::DType, int32_t>;
  static_assert((!IsDense16 && Elements == 32) ||
                    ((IsDense16 || IsInteger32) && Elements == 64),
                "element TPARTVIEW part size must match its dtype profile");
  static_assert(
      (!IsDense16 &&
       (Parent::Numel == 128 || Parent::Numel == 512 ||
        ((IsInteger32 || Parent::IsElement64Type) && Elements == 32 &&
         (Parent::Numel == 64 || Parent::Numel == 256 ||
          Parent::Numel == 1024 ||
          (IsInteger32 && Parent::Numel == 2048))) ||
        (std::is_same_v<typename Parent::DType, float> && Elements == 32 &&
         (Parent::Numel == 2048 || Parent::Numel == 16384)) ||
        (IsInteger32 && Elements == 64 && Parent::Numel == 64))) ||
          (IsDense16 &&
           (Parent::Numel == 64 || Parent::Numel == 128 ||
            Parent::Numel == 512 || Parent::Numel == 2048)),
      "element TPARTVIEW parent is not a supported element profile");
  using SubTile = ElementTile<typename Parent::DType, Elements>;
  constexpr int PartCount = Parent::Numel / Elements;
  return region::BorrowedTileArray<Parent, SubTile, 1, PartCount>(
      parent, valid_elements);
}

template <typename TileType>
  requires(is_element_tile_v<TileType> &&
           std::remove_cvref_t<TileType>::Numel == 32 &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          uint64_t>)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v1;dtype=u64;rows=32;cols=1;layout=cube_m32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_element_tile_v<TileType> &&
           std::remove_cvref_t<TileType>::Numel == 32 &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          int64_t>)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v1;dtype=s64;rows=32;cols=1;layout=cube_m32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_element_tile_v<TileType> &&
           std::remove_cvref_t<TileType>::Numel == 32 &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          uint32_t>)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v1;dtype=u32;rows=32;cols=1;layout=cube_m32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_element_tile_v<TileType> &&
           std::remove_cvref_t<TileType>::Numel == 64 &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          uint32_t>)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v1;dtype=u32;rows=32;cols=2;layout=cube_m32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_element_tile_v<TileType> &&
           std::remove_cvref_t<TileType>::Numel == 64 &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          int32_t>)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v1;dtype=s32;rows=32;cols=2;layout=cube_m32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_element_tile_v<TileType> &&
           std::remove_cvref_t<TileType>::Numel == 64 &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          __half>)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v1;dtype=f16;rows=32;cols=2;layout=cube_m32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_element_tile_v<TileType> &&
           std::remove_cvref_t<TileType>::Numel == 64 &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          uint16_t>)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v1;dtype=u16;rows=32;cols=2;layout=cube_m32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_element_tile_v<TileType> &&
           std::remove_cvref_t<TileType>::Numel == 32 &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          int32_t>)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v1;dtype=s32;rows=32;cols=1;layout=cube_m32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_element_tile_v<TileType> &&
           std::remove_cvref_t<TileType>::Numel == 32 &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          float>)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v1;dtype=f32;rows=32;cols=1;layout=cube_m32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

// Full-row NORM views expose the existing contiguous carrier directly. They
// are not partitions and do not reinterpret NORM storage as CUBE_M32.  The v2
// metadata keeps physical carrier geometry separate from the legal scalar
// loop domain so a 128-column tail view never activates lanes 32..127.
template <typename TileType>
  requires(is_tile_data_v<TileType> &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          uint32_t> &&
           std::remove_cvref_t<TileType>::Loc == Location::Vec &&
           std::remove_cvref_t<TileType>::BFractal == BLayout::RowMajor &&
           std::remove_cvref_t<TileType>::Rows == 1 &&
           std::remove_cvref_t<TileType>::Cols == 128 &&
           std::remove_cvref_t<TileType>::ValidRow == 1 &&
           std::remove_cvref_t<TileType>::ValidCol == 128)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v2;dtype=u32;layout=norm;physical_rows=1;physical_cols=128;valid_rows=1;valid_cols=128")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_tile_data_v<TileType> &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          uint32_t> &&
           std::remove_cvref_t<TileType>::Loc == Location::Vec &&
           std::remove_cvref_t<TileType>::BFractal == BLayout::RowMajor &&
           std::remove_cvref_t<TileType>::Rows == 1 &&
           std::remove_cvref_t<TileType>::Cols == 128 &&
           std::remove_cvref_t<TileType>::ValidRow == 1 &&
           std::remove_cvref_t<TileType>::ValidCol == 32)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v2;dtype=u32;layout=norm;physical_rows=1;physical_cols=128;valid_rows=1;valid_cols=32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

template <typename TileType>
  requires(is_tile_data_v<TileType> &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          uint32_t> &&
           std::remove_cvref_t<TileType>::Loc == Location::Vec &&
           std::remove_cvref_t<TileType>::BFractal == BLayout::RowMajor &&
           std::remove_cvref_t<TileType>::Rows == 1 &&
           std::remove_cvref_t<TileType>::Cols == 2048 &&
           std::remove_cvref_t<TileType>::ValidRow == 1 &&
           std::remove_cvref_t<TileType>::ValidCol == 2048)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v2;dtype=u32;layout=norm;physical_rows=1;physical_cols=2048;valid_rows=1;valid_cols=2048")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
  return tile.data();
}

// Exact FP32 NORM views used by scalar element loops over ordinary contiguous
// RowMajor storage. The matrix profile exposes all 16x32 physical elements;
// the row profile exposes one 1x32 Tile. Neither view changes layout or storage.
template <typename TileType>
  requires(is_tile_data_v<TileType> &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          float> &&
           std::remove_cvref_t<TileType>::Loc == Location::Vec &&
           std::remove_cvref_t<TileType>::BFractal == BLayout::RowMajor &&
           std::remove_cvref_t<TileType>::SFractal == SLayout::NoneBox &&
           std::remove_cvref_t<TileType>::Rows == 16 &&
           std::remove_cvref_t<TileType>::Cols == 32 &&
           std::remove_cvref_t<TileType>::ValidRow == 16 &&
           std::remove_cvref_t<TileType>::ValidCol == 32)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v2;dtype=f32;layout=norm;physical_rows=16;physical_cols=32;valid_rows=16;valid_cols=32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
#if defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)
  using LogicalCarrier =
      linx_tile_carrier<std::remove_cvref_t<TileType>::LogicalTileBytes, float>;
  return reinterpret_cast<typename LogicalCarrier::RegisterType &>(tile.data());
#else
  return tile.data();
#endif
}

template <typename TileType>
  requires(is_tile_data_v<TileType> &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          float> &&
           std::remove_cvref_t<TileType>::Loc == Location::Vec &&
           std::remove_cvref_t<TileType>::BFractal == BLayout::RowMajor &&
           std::remove_cvref_t<TileType>::SFractal == SLayout::NoneBox &&
           std::remove_cvref_t<TileType>::Rows == 16 &&
           std::remove_cvref_t<TileType>::Cols == 32 &&
           std::remove_cvref_t<TileType>::ValidRow == 16 &&
           std::remove_cvref_t<TileType>::ValidCol == 32)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v2;dtype=f32;layout=norm;physical_rows=16;physical_cols=32;valid_rows=16;valid_cols=32")))
#endif
decltype(auto) TPARTELEMENT(const TileType &tile) {
#if defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)
  using LogicalCarrier =
      linx_tile_carrier<std::remove_cvref_t<TileType>::LogicalTileBytes, float>;
  return reinterpret_cast<const typename LogicalCarrier::RegisterType &>(
      tile.data());
#else
  return tile.data();
#endif
}

template <typename TileType>
  requires(is_tile_data_v<TileType> &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          float> &&
           std::remove_cvref_t<TileType>::Loc == Location::Vec &&
           std::remove_cvref_t<TileType>::BFractal == BLayout::RowMajor &&
           std::remove_cvref_t<TileType>::SFractal == SLayout::NoneBox &&
           std::remove_cvref_t<TileType>::Rows == 1 &&
           std::remove_cvref_t<TileType>::Cols == 32 &&
           std::remove_cvref_t<TileType>::ValidRow == 1 &&
           std::remove_cvref_t<TileType>::ValidCol == 32)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v2;dtype=f32;layout=norm;physical_rows=1;physical_cols=32;valid_rows=1;valid_cols=32")))
#endif
decltype(auto) TPARTELEMENT(TileType &tile) {
#if defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)
  using LogicalCarrier =
      linx_tile_carrier<std::remove_cvref_t<TileType>::LogicalTileBytes, float>;
  return reinterpret_cast<typename LogicalCarrier::RegisterType &>(tile.data());
#else
  return tile.data();
#endif
}

template <typename TileType>
  requires(is_tile_data_v<TileType> &&
           std::is_same_v<typename std::remove_cvref_t<TileType>::DType,
                          float> &&
           std::remove_cvref_t<TileType>::Loc == Location::Vec &&
           std::remove_cvref_t<TileType>::BFractal == BLayout::RowMajor &&
           std::remove_cvref_t<TileType>::SFractal == SLayout::NoneBox &&
           std::remove_cvref_t<TileType>::Rows == 1 &&
           std::remove_cvref_t<TileType>::Cols == 32 &&
           std::remove_cvref_t<TileType>::ValidRow == 1 &&
           std::remove_cvref_t<TileType>::ValidCol == 32)
#if defined(__clang__) && defined(__linx)
__attribute__((always_inline, annotate("pto.element.view:v2;dtype=f32;layout=norm;physical_rows=1;physical_cols=32;valid_rows=1;valid_cols=32")))
#endif
decltype(auto) TPARTELEMENT(const TileType &tile) {
#if defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)
  using LogicalCarrier =
      linx_tile_carrier<std::remove_cvref_t<TileType>::LogicalTileBytes, float>;
  return reinterpret_cast<const typename LogicalCarrier::RegisterType &>(
      tile.data());
#else
  return tile.data();
#endif
}



template <typename Parent, typename SubTile, int Rows, int Cols>
auto TASSEMBLY(region::TileArray<SubTile, Rows, Cols> &&array) -> Parent {
  static_assert(std::is_same_v<typename Parent::DType,
                               typename SubTile::DType>,
                "TASSEMBLY parent and fragment dtypes must match");
  static_assert(Parent::Loc == SubTile::Loc,
                "TASSEMBLY parent and fragment locations must match");
  static_assert(Parent::BFractal == SubTile::BFractal,
                "TASSEMBLY parent and fragment layouts must match");
  static_assert(Parent::SFractal == SubTile::SFractal,
                "TASSEMBLY parent and fragment storage layouts must match");
  static_assert(Parent::Rows == Rows * SubTile::Rows &&
                    Parent::Cols == Cols * SubTile::Cols,
                "TASSEMBLY parent and fragment shapes do not match");
  static_assert(Parent::ValidRow == Rows * SubTile::ValidRow &&
                    Parent::ValidCol == Cols * SubTile::ValidCol,
                "TASSEMBLY parent and fragment valid shapes do not match");
  static_assert(Parent::LogicalTileBytes ==
                    SubTile::LogicalTileBytes * Rows * Cols,
                "TASSEMBLY parent and fragment coverage do not match");
  Parent result;
#if defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)
  static_assert(std::is_same_v<typename Parent::TileDType,
                               typename region::TileArray<SubTile, Rows,
                                                          Cols>::ParentCarrier>,
                "TASSEMBLY parent and TileArray carriers must match");
  result.assignData(array.carrier());
#endif
  return result;
}

} // namespace pto

#endif
