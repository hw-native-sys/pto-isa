#ifndef PTO_PREDICATE_HPP
#define PTO_PREDICATE_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include "pto/npu/linx/impl/common/pto_indexed_range.hpp"

namespace pto {
namespace predicate {

template <typename Basis> class Predicate;

namespace detail {
template <typename Basis>
__attribute__((always_inline)) inline decltype(auto)
atomic_tstore_carrier(const Predicate<Basis> &);
} // namespace detail

template <typename Basis, int Columns = Basis::ValidCol>
using LogicalBasis = Tile<Location::Vec, uint8_t,
                          Basis::BFractal == BLayout::CubeM16 ? 16 : 32,
                          Columns, Basis::BFractal, Basis::ValidRow, Columns>;

template <unsigned Operation, typename Left, typename Right>
__attribute__((always_inline)) inline auto
logical(const Predicate<Left> &, const Predicate<Right> &)
    -> Predicate<LogicalBasis<Left>>;

template <unsigned OffsetCells, int Columns, typename Basis>
__attribute__((always_inline)) inline auto subview(const Predicate<Basis> &)
    -> Predicate<Tile<Location::Vec, typename Basis::DType,
                      Basis::BFractal == BLayout::CubeM16 ? 16 : 32,
                      Columns, Basis::BFractal, Basis::ValidRow, Columns>>;

template <typename Left, typename Right>
__attribute__((always_inline)) inline auto
concatenate(const Predicate<Left> &, const Predicate<Right> &)
    -> Predicate<LogicalBasis<Left, Left::ValidCol + Right::ValidCol>>;

template <unsigned Operation, typename Basis, typename Numeric>
__attribute__((always_inline)) inline auto
logical_numeric(const Predicate<Basis> &, const Numeric &)
    -> LogicalBasis<Basis>;

template <typename Basis, typename Value>
__attribute__((always_inline)) inline void
select(Value &, const Predicate<Basis> &, const Value &, const Value &);

template <typename Basis, typename Value>
__attribute__((always_inline)) inline void
select_scalar(Value &, const Predicate<Basis> &, const Value &,
              typename Value::DType);

template <CmpMode Mode, typename Basis>
__attribute__((always_inline)) inline Predicate<Basis>
compare_scalar(Basis &source, typename Basis::DType scalar);

template <typename Basis> class Predicate {
  static_assert(Basis::Loc == Location::Vec &&
                    (Basis::BFractal == BLayout::CubeM32 ||
                     Basis::BFractal == BLayout::CubeM16),
                "Predicate requires a Local VEC CUBE_M16/M32 basis");
  static_assert(Basis::ValidRow > 0 && Basis::ValidCol > 0,
                "Predicate currently requires static valid dimensions");
  static constexpr int StorageRows =
      Basis::BFractal == BLayout::CubeM16 ? 16 : 32;
  static constexpr int MinimumColumns = 128 / StorageRows;
  static constexpr int StorageColumns = Basis::Cols < MinimumColumns
                                           ? MinimumColumns : Basis::Cols;
  using Storage = Tile<Location::Vec, uint8_t, StorageRows, StorageColumns,
                       Basis::BFractal, Basis::ValidRow, Basis::ValidCol>;
  typename Storage::TileRegisterType Carrier;

  Predicate() = default;
  template <CmpMode Mode, typename Source>
  friend __attribute__((always_inline)) inline Predicate<Source>
  compare_scalar(Source &, typename Source::DType);
  template <unsigned Operation, typename Left, typename Right>
  friend __attribute__((always_inline)) inline auto
  logical(const Predicate<Left> &, const Predicate<Right> &)
      -> Predicate<LogicalBasis<Left>>;
  template <unsigned OffsetCells, int Columns, typename Source>
  friend __attribute__((always_inline)) inline auto
  subview(const Predicate<Source> &)
      -> Predicate<Tile<Location::Vec, typename Source::DType,
                        Source::BFractal == BLayout::CubeM16 ? 16 : 32,
                        Columns, Source::BFractal, Source::ValidRow, Columns>>;
  template <typename Left, typename Right>
  friend __attribute__((always_inline)) inline auto
  concatenate(const Predicate<Left> &, const Predicate<Right> &)
      -> Predicate<LogicalBasis<Left, Left::ValidCol + Right::ValidCol>>;
  template <unsigned Operation, typename Source, typename Numeric>
  friend __attribute__((always_inline)) inline auto
  logical_numeric(const Predicate<Source> &, const Numeric &)
      -> LogicalBasis<Source>;
  template <typename Source, typename Value>
  friend __attribute__((always_inline)) inline void
  select(Value &, const Predicate<Source> &, const Value &, const Value &);
  template <typename Source, typename Value>
  friend __attribute__((always_inline)) inline void
  select_scalar(Value &, const Predicate<Source> &, const Value &,
                typename Value::DType);
  template <typename Source, typename Index, typename Value, typename Global>
  friend __attribute__((always_inline)) inline void
  scatter(Global &, const Index &, const Value &, const Predicate<Source> &);
  template <typename Source, typename Index, typename Value, typename Global>
  friend __attribute__((always_inline)) inline void
  scatter(Global &, const Index &, const Value &, const Predicate<Source> &,
          IndexedByteRange);
  template <bool Invert, typename Source, typename Index, typename Value,
            typename Global>
  friend __attribute__((always_inline)) inline void
  atomic_add(Global &, const Index &, const Value &, const Predicate<Source> &);
  template <bool Invert, typename Source, typename Index, typename Value,
            typename Global>
  friend __attribute__((always_inline)) inline void
  atomic_add(Global &, const Index &, const Value &, const Predicate<Source> &,
             IndexedByteRange);
  template <typename Source>
  friend __attribute__((always_inline)) inline decltype(auto)
  detail::atomic_tstore_carrier(const Predicate<Source> &);

public:
  using BasisTile = Basis;
};

namespace detail {
template <typename Basis>
__attribute__((always_inline)) inline decltype(auto)
atomic_tstore_carrier(const Predicate<Basis> &predicate) {
  return (predicate.Carrier);
}
} // namespace detail

#define PTO_PREDICATE_LAYOUT_ASM                                              \
  ".if %c[Layout] == 31\nB.DATR CUBE_M16, DTYPE_NONE, Zero\n"                 \
  ".elseif %c[Layout] == 29\nB.DATR CUBE_M32, DTYPE_NONE, Zero\n.endif\n"

template <CmpMode Mode, typename Basis>
__attribute__((always_inline)) inline Predicate<Basis>
compare_scalar(Basis &source, typename Basis::DType scalar) {
  static_assert(Mode == CmpMode::EQ || Mode == CmpMode::NE ||
                    Mode == CmpMode::LT || Mode == CmpMode::GT ||
                    Mode == CmpMode::LE || Mode == CmpMode::GE,
                "Predicate comparison mode is reserved");
  static_assert(std::is_same_v<typename Basis::DType, float> ||
                    std::is_same_v<typename Basis::DType, uint32_t> ||
                    std::is_same_v<typename Basis::DType, int32_t>,
                "Predicate comparison currently supports FP32/S32/U32");
  Predicate<Basis> result;
  asm volatile(
      "BSTART.TEPL 45, %D[Type]\n"
      ".if %c[Layout] == 31\n"
      ".if %c[Mode] == 0\nB.DATR CUBE_M16, DTYPE_NONE, Zero, EQ, RNONE\n"
      ".elseif %c[Mode] == 1\nB.DATR CUBE_M16, DTYPE_NONE, Zero, NE, RNONE\n"
      ".elseif %c[Mode] == 2\nB.DATR CUBE_M16, DTYPE_NONE, Zero, LT, RNONE\n"
      ".elseif %c[Mode] == 3\nB.DATR CUBE_M16, DTYPE_NONE, Zero, GT, RNONE\n"
      ".elseif %c[Mode] == 4\nB.DATR CUBE_M16, DTYPE_NONE, Zero, LE, RNONE\n"
      ".else\nB.DATR CUBE_M16, DTYPE_NONE, Zero, GE, RNONE\n.endif\n"
      ".elseif %c[Layout] == 29\n"
      ".if %c[Mode] == 0\nB.DATR CUBE_M32, DTYPE_NONE, Zero, EQ, RNONE\n"
      ".elseif %c[Mode] == 1\nB.DATR CUBE_M32, DTYPE_NONE, Zero, NE, RNONE\n"
      ".elseif %c[Mode] == 2\nB.DATR CUBE_M32, DTYPE_NONE, Zero, LT, RNONE\n"
      ".elseif %c[Mode] == 3\nB.DATR CUBE_M32, DTYPE_NONE, Zero, GT, RNONE\n"
      ".elseif %c[Mode] == 4\nB.DATR CUBE_M32, DTYPE_NONE, Zero, LE, RNONE\n"
      ".else\nB.DATR CUBE_M32, DTYPE_NONE, Zero, GE, RNONE\n.endif\n.endif\n"
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[PhysicalColumns], ->LB2\n"
      "B.IOT %[Source], mask=1111, last, ->%[Result]<%Z[Size]>\n"
      "B.IOR [%[Scalar]], []\n"
      : [Result] "=Tr"(result.Carrier)
      : [Source] "Tr"(source.data()), [Scalar] "r"(scalar),
        [Type] "i"(type_traits<typename Basis::DType>::TypeCode),
        [Mode] "i"(static_cast<int>(Mode)),
        [Layout] "i"(Basis::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Rows] "i"(Basis::ValidRow), [Columns] "i"(Basis::ValidCol),
        [PhysicalColumns] "i"(Basis::Cols),
        [Size] "i"(tile_type_traits<typename Predicate<Basis>::Storage::TileDType>::TilesizeCode));
  return result;
}

template <typename Basis, typename Index, typename Value>
constexpr void validate_consumer() {
  static_assert(Basis::BFractal == BLayout::CubeM32,
                "Predicate indexed memory currently requires CUBE_M32");
  static_assert(Index::Loc == Location::Vec && Value::Loc == Location::Vec,
                "Predicate consumers require Local VEC operands");
  static_assert(Index::BFractal == Basis::BFractal &&
                    Value::BFractal == Basis::BFractal,
                "Predicate consumer layout must match producer layout");
  static_assert(Index::ValidRow == Basis::ValidRow &&
                    Index::ValidCol == Basis::ValidCol &&
                    Value::ValidRow == Basis::ValidRow &&
                    Value::ValidCol == Basis::ValidCol,
                "Predicate consumer valid shape must match producer shape");
  static_assert(std::is_same_v<typename Index::DType, uint64_t> ||
                    std::is_same_v<typename Index::DType, int64_t>,
                "Predicate indices must be S64/U64 byte displacements");
  static_assert(std::is_same_v<typename Value::DType, uint32_t> ||
                    std::is_same_v<typename Value::DType, int32_t> ||
                    std::is_same_v<typename Value::DType, float>,
                "Predicate consumers currently support FP32/S32/U32 values");
  static_assert(Index::Cols >= Basis::ValidCol &&
                    Value::Cols >= Basis::ValidCol,
                "Predicate operands must cover the complete valid region");
}

template <typename Basis, typename Index, typename Value, typename Global>
__attribute__((always_inline)) inline void
scatter(Global &destination, const Index &offsets, const Value &values,
        const Predicate<Basis> &active) {
  validate_consumer<Basis, Index, Value>();
  static_assert(std::is_same_v<typename Global::DType, typename Value::DType>,
                "Predicate destination element type must match values");
  asm volatile(
      "BSTART.TLSU MSCATTER.MASK, %D[Type]\n"
      PTO_PREDICATE_LAYOUT_ASM
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[PhysicalColumns], ->LB2\n"
      "B.IOT %[Values], %[Offsets], mask=1111\n"
      "B.IOT %[Predicate], mask=1111, last\n"
      "B.IOR [%[Base]], []\n"
      :
      : [Values] "Tr"(values.data()), [Offsets] "Tr"(offsets.data()),
        [Predicate] "Tr"(active.Carrier), [Base] "r"(destination.data()),
        [Type] "i"(type_traits<typename Value::DType>::TypeCode),
        [Layout] "i"(Basis::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Columns] "i"(Basis::ValidCol), [Rows] "i"(Basis::ValidRow),
        [PhysicalColumns] "i"(Value::Cols)
      : "memory");
}

template <typename Basis, typename Index, typename Value, typename Global>
__attribute__((always_inline)) inline void
scatter(Global &destination, const Index &offsets, const Value &values,
        const Predicate<Basis> &active, IndexedByteRange range) {
  validate_consumer<Basis, Index, Value>();
  static_assert(std::is_same_v<typename Global::DType, typename Value::DType>,
                "Predicate destination element type must match values");
  asm volatile(
      "BSTART.TLSU MSCATTER.MASK, %D[Type]\n"
      PTO_PREDICATE_LAYOUT_ASM
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[PhysicalColumns], ->LB2\n"
      "B.IOT %[Values], %[Offsets], mask=1111\n"
      "B.IOT %[Predicate], mask=1111, last\n"
      "B.IOR [%[Base],%[RangeMin],%[RangeMax]], []\n"
      :
      : [Values] "Tr"(values.data()), [Offsets] "Tr"(offsets.data()),
        [Predicate] "Tr"(active.Carrier), [RangeMin] "R"(range.min), [RangeMax] "r"(range.max),
        [Base] "r"(destination.data()),
        [Type] "i"(type_traits<typename Value::DType>::TypeCode),
        [Layout] "i"(Basis::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Columns] "i"(Basis::ValidCol), [Rows] "i"(Basis::ValidRow),
        [PhysicalColumns] "i"(Value::Cols)
      : "memory");
}

template <bool Invert = false, typename Basis, typename Index, typename Value,
          typename Global>
__attribute__((always_inline)) inline void
atomic_add(Global &destination, const Index &offsets, const Value &values,
           const Predicate<Basis> &active) {
  validate_consumer<Basis, Index, Value>();
  static_assert(std::is_same_v<typename Global::DType, typename Value::DType>,
                "Predicate destination element type must match values");
  ew_mscatter_add_predicate(
      Basis::ValidRow, Basis::ValidCol, Value::Cols,
      type_traits<typename Value::DType>::TypeCode,
      Basis::BFractal == BLayout::CubeM16 ? 31 : 29, destination.data(),
      values.data(), offsets.data(), active.Carrier, Invert);
}

template <bool Invert = false, typename Basis, typename Index, typename Value,
          typename Global>
__attribute__((always_inline)) inline void
atomic_add(Global &destination, const Index &offsets, const Value &values,
           const Predicate<Basis> &active, IndexedByteRange range) {
  validate_consumer<Basis, Index, Value>();
  static_assert(std::is_same_v<typename Global::DType, typename Value::DType>,
                "Predicate destination element type must match values");
  ew_mscatter_add_predicate_range(
      Basis::ValidRow, Basis::ValidCol, Value::Cols,
      type_traits<typename Value::DType>::TypeCode,
      Basis::BFractal == BLayout::CubeM16 ? 31 : 29, destination.data(),
      values.data(), offsets.data(), active.Carrier, Invert, range.min, range.max);
}

template <typename Basis, typename Value>
constexpr void validate_selection() {
  static_assert(Value::Loc == Location::Vec &&
                    Value::BFractal == Basis::BFractal,
                "Predicate selection requires matching Local CUBE layouts");
  static_assert(Value::ValidRow == Basis::ValidRow &&
                    Value::ValidCol == Basis::ValidCol,
                "Predicate selection valid shape must match");
  static_assert(std::is_same_v<typename Value::DType, uint32_t> ||
                    std::is_same_v<typename Value::DType, int32_t> ||
                    std::is_same_v<typename Value::DType, float>,
                "Predicate selection supports FP32/S32/U32 values");
}

template <unsigned OffsetCells, int Columns, typename Basis>
__attribute__((always_inline)) inline auto
subview(const Predicate<Basis> &source)
    -> Predicate<Tile<Location::Vec, typename Basis::DType,
                      Basis::BFractal == BLayout::CubeM16 ? 16 : 32,
                      Columns, Basis::BFractal, Basis::ValidRow, Columns>> {
  using ResultBasis = Tile<Location::Vec, typename Basis::DType,
                           Basis::BFractal == BLayout::CubeM16 ? 16 : 32,
                           Columns, Basis::BFractal, Basis::ValidRow, Columns>;
  Predicate<ResultBasis> result;
  constexpr unsigned CellColumns = Basis::BFractal == BLayout::CubeM16 ? 8 : 4;
  static_assert(Columns > 0 && Columns % CellColumns == 0,
                "Predicate subview must cover complete CELL columns");
  static_assert(OffsetCells * CellColumns + Columns <= Basis::ValidCol,
                "Predicate subview must fit the source valid region");
  static_assert(decltype(result)::Storage::TilesizeCode <= 10,
                "Predicate subview capacity must be at most 64KB");
  asm volatile(
      "BSTART.TLSU TMOV, U8\n"
      PTO_PREDICATE_LAYOUT_ASM
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[PhysicalColumns], ->LB2\n"
      "B.IOT %[Source], mask=1111, last, ->%[Result]<%Z[Size]>\n"
      "B.SUBVIEW 0, zero, %c[Offset], %c[Size]\n"
      : [Result] "=&Tr"(result.Carrier)
      : [Source] "Tr"(source.Carrier), [Offset] "i"(OffsetCells),
        [Layout] "i"(Basis::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Rows] "i"(Basis::ValidRow), [Columns] "i"(Columns),
        [PhysicalColumns] "i"(decltype(result)::StorageColumns),
        [Size] "i"(decltype(result)::Storage::TilesizeCode));
  return result;
}

template <typename Left, typename Right>
__attribute__((always_inline)) inline auto
concatenate(const Predicate<Left> &left, const Predicate<Right> &right)
    -> Predicate<LogicalBasis<Left, Left::ValidCol + Right::ValidCol>> {
  static_assert(Left::BFractal == Right::BFractal &&
                    Left::ValidRow == Right::ValidRow,
                "Predicate assembly layouts and valid rows must match");
  static_assert(std::is_same_v<typename Left::DType, uint8_t> &&
                    std::is_same_v<typename Right::DType, uint8_t>,
                "Predicate assembly requires U8-basis logical results");
  static_assert(Left::ValidCol == Right::ValidCol &&
                    Left::ValidCol == Predicate<Left>::StorageColumns &&
                    Right::ValidCol == Predicate<Right>::StorageColumns,
                "Predicate assembly requires equal complete writer regions");
  static_assert(Left::ValidRow ==
                    (Left::BFractal == BLayout::CubeM16 ? 16 : 32),
                "Predicate assembly requires full physical writer rows");
  static_assert(Predicate<Left>::Storage::TilesizeCode <= 10,
                "Predicate assembly writer capacity must be at most 64KB");
  Predicate<LogicalBasis<Left, Left::ValidCol + Right::ValidCol>> result;
  asm volatile(
      "BSTART.TEPL TAND, U8\n"
      PTO_PREDICATE_LAYOUT_ASM
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[Columns], ->LB2\n"
      "B.IOT %[Left], %[Left], mask=1111, last, ->%[Result]<%Z[Size]>\n"
      "B.ASSEMBLE %c[Init], %c[Last], zero, 0, %c[WriterSize]\n"
      : [Result] "=&Tr"(result.Carrier)
      : [Left] "Tr"(left.Carrier),
        [Layout] "i"(Left::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Rows] "i"(Left::ValidRow), [Columns] "i"(Left::ValidCol),
        [Size] "i"(decltype(result)::Storage::TilesizeCode),
        [Init] "i"(1), [Last] "i"(0),
        [WriterSize] "i"(Predicate<Left>::Storage::TilesizeCode));
  asm volatile(
      "BSTART.TEPL TAND, U8\n"
      PTO_PREDICATE_LAYOUT_ASM
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[Columns], ->LB2\n"
      "B.IOT %[Right], %[Right], mask=1111, last\n"
      "B.IOT %[Parent], mask=1111\n"
      "B.ASSEMBLE %c[Init], %c[Last], zero, %c[Offset], %c[WriterSize]\n"
      :
      : [Parent] "Tr"(result.Carrier), [Right] "Tr"(right.Carrier),
        [Layout] "i"(Left::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Rows] "i"(Right::ValidRow), [Columns] "i"(Right::ValidCol),
        [Init] "i"(0), [Last] "i"(1),
        [Offset] "i"(Predicate<Left>::Storage::StorageBytes / 128),
        [WriterSize] "i"(Predicate<Right>::Storage::TilesizeCode));
  return result;
}

template <unsigned Operation, typename Left, typename Right>
__attribute__((always_inline)) inline auto
logical(const Predicate<Left> &left, const Predicate<Right> &right)
    -> Predicate<LogicalBasis<Left>> {
  static_assert(Operation == 6 || Operation == 7 || Operation == 8,
                "Predicate logical operation must be AND, OR or XOR");
  static_assert(Left::BFractal == Right::BFractal &&
                    Left::ValidRow == Right::ValidRow &&
                    Left::ValidCol == Right::ValidCol,
                "Predicate logical layouts and valid shapes must match");
  static_assert(Predicate<Left>::Storage::StorageBytes ==
                    Predicate<Right>::Storage::StorageBytes &&
                    Predicate<Left>::Storage::StorageBytes ==
                        Predicate<LogicalBasis<Left>>::Storage::StorageBytes,
                "Predicate logical physical shapes must match the result");
  Predicate<LogicalBasis<Left>> result;
  asm volatile(
      "BSTART.TEPL %c[Operation], U8\n"
      PTO_PREDICATE_LAYOUT_ASM
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[PhysicalColumns], ->LB2\n"
      "B.IOT %[Left], %[Right], mask=1111, last, ->%[Result]<%Z[Size]>\n"
      : [Result] "=Tr"(result.Carrier)
      : [Left] "Tr"(left.Carrier), [Right] "Tr"(right.Carrier),
        [Operation] "i"(Operation),
        [Layout] "i"(Left::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Rows] "i"(Left::ValidRow), [Columns] "i"(Left::ValidCol),
        [PhysicalColumns] "i"(decltype(result)::StorageColumns),
        [Size] "i"(decltype(result)::Storage::TilesizeCode));
  return result;
}

template <typename Left, typename Right>
__attribute__((always_inline)) inline auto
logical_and(const Predicate<Left> &left, const Predicate<Right> &right) {
  return logical<6>(left, right);
}

template <typename Left, typename Right>
__attribute__((always_inline)) inline auto
logical_or(const Predicate<Left> &left, const Predicate<Right> &right) {
  return logical<7>(left, right);
}

template <typename Left, typename Right>
__attribute__((always_inline)) inline auto
logical_xor(const Predicate<Left> &left, const Predicate<Right> &right) {
  return logical<8>(left, right);
}

template <unsigned Operation, typename Basis, typename Numeric>
__attribute__((always_inline)) inline auto
logical_numeric(const Predicate<Basis> &active, const Numeric &numeric)
    -> LogicalBasis<Basis> {
  static_assert(Operation == 6 || Operation == 7 || Operation == 8,
                "Predicate logical operation must be AND, OR or XOR");
  static_assert(Numeric::Loc == Location::Vec &&
                    Numeric::BFractal == Basis::BFractal &&
                    Numeric::ValidRow == Basis::ValidRow &&
                    Numeric::ValidCol == Basis::ValidCol,
                "Mixed logical layouts and valid shapes must match");
  static_assert(std::is_same_v<typename Numeric::DType, uint8_t> ||
                    std::is_same_v<typename Numeric::DType, int8_t>,
                "Mixed logical input must be Numeric S8/U8");
  static_assert(Numeric::StorageBytes == Predicate<Basis>::Storage::StorageBytes &&
                    Numeric::StorageBytes == LogicalBasis<Basis>::StorageBytes,
                "Mixed logical physical shapes must match");
  LogicalBasis<Basis> result;
  asm volatile(
      "BSTART.TEPL %c[Operation], U8\n"
      PTO_PREDICATE_LAYOUT_ASM
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[PhysicalColumns], ->LB2\n"
      "B.IOT %[Predicate], %[Numeric], mask=1111, last, ->%[Result]<%Z[Size]>\n"
      : [Result] "=Tr"(result.data())
      : [Predicate] "Tr"(active.Carrier), [Numeric] "Tr"(numeric.data()),
        [Operation] "i"(Operation),
        [Layout] "i"(Basis::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Rows] "i"(Basis::ValidRow), [Columns] "i"(Basis::ValidCol),
        [PhysicalColumns] "i"(LogicalBasis<Basis>::Cols),
        [Size] "i"(LogicalBasis<Basis>::TilesizeCode));
  return result;
}

template <typename Basis, typename Value>
__attribute__((always_inline)) inline void
select(Value &destination, const Predicate<Basis> &active,
       const Value &when_true, const Value &when_false) {
  validate_selection<Basis, Value>();
  asm volatile(
      "BSTART.TEPL TSEL, %D[Type]\n"
      PTO_PREDICATE_LAYOUT_ASM
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[PhysicalColumns], ->LB2\n"
      "B.IOT %[Predicate], %[True], mask=1111\n"
      "B.IOT %[False], mask=1111, last, ->%[Result]<%Z[Size]>\n"
      : [Result] "=Tr"(destination.data())
      : [Predicate] "Tr"(active.Carrier), [True] "Tr"(when_true.data()),
        [False] "Tr"(when_false.data()),
        [Type] "i"(type_traits<typename Value::DType>::TypeCode),
        [Layout] "i"(Basis::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Rows] "i"(Basis::ValidRow), [Columns] "i"(Basis::ValidCol),
        [PhysicalColumns] "i"(Value::Cols), [Size] "i"(Value::TilesizeCode));
}

template <typename Basis, typename Value>
__attribute__((always_inline)) inline void
select_scalar(Value &destination, const Predicate<Basis> &active,
              const Value &when_true, typename Value::DType when_false) {
  validate_selection<Basis, Value>();
  asm volatile(
      "BSTART.TEPL TSELS, %D[Type]\n"
      PTO_PREDICATE_LAYOUT_ASM
      "B.DIM zero, %c[Columns], ->LB0\n"
      "B.DIM zero, %c[Rows], ->LB1\n"
      "B.DIM zero, %c[PhysicalColumns], ->LB2\n"
      "B.IOT %[Predicate], %[True], mask=1111, last, ->%[Result]<%Z[Size]>\n"
      "B.IOR [%[Scalar]], []\n"
      : [Result] "=Tr"(destination.data())
      : [Predicate] "Tr"(active.Carrier), [True] "Tr"(when_true.data()),
        [Scalar] "r"(when_false),
        [Type] "i"(type_traits<typename Value::DType>::TypeCode),
        [Layout] "i"(Basis::BFractal == BLayout::CubeM16 ? 31 : 29),
        [Rows] "i"(Basis::ValidRow), [Columns] "i"(Basis::ValidCol),
        [PhysicalColumns] "i"(Value::Cols), [Size] "i"(Value::TilesizeCode));
}

#undef PTO_PREDICATE_LAYOUT_ASM

}
}

#endif
