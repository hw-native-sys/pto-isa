#ifndef PTO_LOCAL_CUBE_ASSEMBLY_HPP
#define PTO_LOCAL_CUBE_ASSEMBLY_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include <array>
#include <tuple>
#include <utility>

namespace pto {

namespace detail { struct LocalCubeAssemblyAccess; }

// Static Local M32 destination assembly. The Parent supplies allocation only;
// writer descriptors and LAST's aggregate descriptor have independent geometry.
// Contract: PTO-spec 4f5a0c23, PTO-B-ASSEMBLE-CUBE-PARENT-GEOMETRY-001.
template <typename Parent, typename... Writers>
class LocalCubeAssembly;

template <typename Parent, int Columns, int ValidRows, int ValidColumns>
class LocalCubeAssemblyResult {
  template <typename, typename...> friend class LocalCubeAssembly;
  Parent storage_;
  __attribute__((always_inline)) explicit LocalCubeAssemblyResult(
      const Parent &storage)
      : storage_(storage) {}

public:
  using DType = typename Parent::DType;
  static constexpr int PhysicalRows = 32;
  static constexpr int PhysicalCols = Columns;
  static constexpr int ValidRow = ValidRows;
  static constexpr int ValidCol = ValidColumns;
  static constexpr int CapacityBytes = Parent::LogicalTileBytes;
  static constexpr int PhysicalBytes = 32 * Columns * tile_element_bits_v<DType> / 8;
#ifdef __linx
  __attribute__((always_inline)) const auto &data() const {
    return storage_.data();
  }
#endif
};

template <typename Assembly, std::size_t Index>
class LocalCubeAssemblyWriter {
  friend Assembly;
  Assembly &assembly_;
  explicit LocalCubeAssemblyWriter(Assembly &assembly) : assembly_(assembly) {}

public:
  using Writer = typename Assembly::template WriterType<Index>;
  static constexpr bool INIT = Index == 0;
  static constexpr bool LAST = Index + 1 == Assembly::WriterCount;
  static constexpr int ParentSizeCode = Assembly::ParentSizeCode;
  static constexpr int WriterSizeCode = Writer::TilesizeCode;
  static constexpr int OffsetCells = Assembly::template offset<Index>();
  LocalCubeAssemblyWriter(const LocalCubeAssemblyWriter &) = delete;
  LocalCubeAssemblyWriter &operator=(const LocalCubeAssemblyWriter &) = delete;

private:
  friend struct detail::LocalCubeAssemblyAccess;
  // Used only by the dedicated TLSU lowering; this carrier is not an is_tile.
  __attribute__((always_inline)) decltype(auto) parent_data() {
    assembly_.template check_writer<Index>();
    return assembly_.parent_.data();
  }
  __attribute__((always_inline)) void complete() {
    assembly_.template check_writer<Index>();
    ++assembly_.next_;
  }
};

template <typename Parent, typename... Writers>
class LocalCubeAssembly {
  template <typename, std::size_t> friend class LocalCubeAssemblyWriter;
  static_assert(sizeof...(Writers) >= 1 && sizeof...(Writers) <= 16,
                "Local M32 assembly requires 1..16 writers");
  static_assert(is_local_tile_v<Parent> && Parent::BFractal == BLayout::CubeM32 &&
                    Parent::IsValidActiveSize && Parent::TilesizeCode <= 10,
                "Local M32 assembly requires a Local M32 parent capacity of 128B..64KiB");
  static_assert(((is_local_tile_v<Writers> &&
                     Writers::BFractal == BLayout::CubeM32 &&
                     Writers::Loc == Parent::Loc &&
                     std::is_same_v<typename Parent::DType,
                                    typename Writers::DType>) && ...),
                "Local M32 assembly writer dtype, location and layout must match parent");
  static_assert(((Writers::Rows == 32 && Writers::ValidRow > 0 &&
                     Writers::ValidRow <= 32 && Writers::ValidCol > 0 &&
                     Writers::ValidCol <= Writers::Cols) && ...),
                "Local M32 assembly requires static valid writer shapes with 32 physical rows");
  static_assert(((Writers::IsValidActiveSize && Writers::TilesizeCode <= 10 &&
                     Writers::CubeRequiredBytes == Writers::LogicalTileBytes &&
                     ((Writers::ValidCol + Writers::CubeCellCols - 1) /
                          Writers::CubeCellCols) * Writers::CubeCellCols ==
                         Writers::CubeStorageCols) && ...),
                "Local M32 assembly writer must have an exact power-of-two CELL envelope");

public:
  static constexpr std::size_t WriterCount = sizeof...(Writers);
  static constexpr int ParentSizeCode = Parent::TilesizeCode;
  template <std::size_t Index>
  using WriterType = std::tuple_element_t<Index, std::tuple<Writers...>>;

private:
  inline static constexpr std::array<int, WriterCount> Cells = {
      (Writers::CubeRequiredBytes / 128)...};
  inline static constexpr std::array<int, WriterCount> Columns = {
      Writers::CubeStorageCols...};
  inline static constexpr std::array<int, WriterCount> ValidColumns = {
      Writers::ValidCol...};
  inline static constexpr std::array<int, WriterCount> ValidRows = {
      Writers::ValidRow...};
  static constexpr int CoveredBytes = (Writers::CubeRequiredBytes + ...);
  static constexpr int FinalColumns = (Writers::CubeStorageCols + ...);
  static constexpr int FinalValidColumns =
      FinalColumns - Columns[WriterCount - 1] + ValidColumns[WriterCount - 1];
  static_assert(CoveredBytes <= Parent::LogicalTileBytes,
                "Local M32 assembly writer coverage exceeds parent capacity");
  static_assert([] {
    int offset = 0;
    for (std::size_t i = 0; i < WriterCount; ++i) {
      if (offset > 2047 ||
          !Parent::CubePhysicalCellRangeComplete(offset, Cells[i])) return false;
      offset += Cells[i];
      if (ValidRows[i] != ValidRows[0]) return false;
      if (i + 1 < WriterCount && ValidColumns[i] != Columns[i]) return false;
    }
    return true;
  }(), "Local M32 assembly requires common valid rows, complete CELL pairs and only a terminal column tail");

  Parent parent_;
  std::size_t next_ = 0;
  template <std::size_t Index>
  __attribute__((always_inline)) void check_writer() const {
    if (next_ != Index) __builtin_trap();
  }

public:
  LocalCubeAssembly() = default;
  LocalCubeAssembly(const LocalCubeAssembly &) = delete;
  LocalCubeAssembly &operator=(const LocalCubeAssembly &) = delete;
  LocalCubeAssembly(LocalCubeAssembly &&) = delete;
  LocalCubeAssembly &operator=(LocalCubeAssembly &&) = delete;

  template <std::size_t Index>
  static constexpr int offset() {
    static_assert(Index < WriterCount, "Local M32 assembly writer index out of range");
    int result = 0;
    for (std::size_t i = 0; i < Index; ++i) result += Cells[i];
    return result;
  }
  template <std::size_t Index>
  __attribute__((always_inline)) auto writer() & {
    static_assert(Index < WriterCount, "Local M32 assembly writer index out of range");
    check_writer<Index>();
    return LocalCubeAssemblyWriter<LocalCubeAssembly, Index>(*this);
  }
  using Result = LocalCubeAssemblyResult<Parent, FinalColumns, ValidRows[0],
                                         FinalValidColumns>;
  __attribute__((always_inline)) Result finish() & {
    if (next_ != WriterCount) __builtin_trap();
    ++next_;
    return Result(parent_);
  }
};

namespace detail {
// Internal lowering bridge. Public writer carriers expose no storage or
// lifecycle mutation; only a producer may complete a slot.
struct LocalCubeAssemblyAccess {
  template <typename Slot>
  __attribute__((always_inline)) static decltype(auto) parent_data(Slot &slot) {
    return slot.parent_data();
  }
  template <typename Slot>
  __attribute__((always_inline)) static void complete(Slot &slot) {
    slot.complete();
  }
};
} // namespace detail

} // namespace pto
#endif
