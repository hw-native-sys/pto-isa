/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/npu/linx/experimental.hpp>
#ifndef PTO_LINX_NATIVE_PTO_TILE_REGION_INLINE_ASM_HPP
#define PTO_LINX_NATIVE_PTO_TILE_REGION_INLINE_ASM_HPP

#include "pto/npu/linx/impl/common/pto_tile_region.hpp"

namespace pto {

#define PTO_REGION_ALWAYS_INLINE __attribute__((always_inline)) inline

template <int Opcode, typename Out, typename Parent, typename SubTile>
PTO_REGION_ALWAYS_INLINE void
pto_region_unary(Out &dst, region::SubTileView<Parent, SubTile> &src) {
  static_assert(range::is_legal_subview_parent_v<Parent>,
                "B.SUBVIEW source must be an assigned Local CUBE or Shared RowMajor "
                "tile layout");
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  const uintptr_t region_base_units = src.GetRangeBase();
  {
    static_assert(SubTile::IsCubeLayout || Opcode == 64,
                  "B.SUBVIEW source layout is not legal for this region op");
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
        "BSTART.TEPL %c8, %D1\n"
        PTO_REGION_ELEMENTWISE_LAYOUT_ASM
        "B.DIM zero, %c3, ->lb0\n"
        "B.DIM zero, %c4, ->lb1\n"
        "B.DIM zero, %c5, ->lb2\n"
        "B.IOT %2, mask=1111, last, ->%0<%Z6>\n"
        "B.SUBVIEW 0, %9, 0, %c7\n"
        : [Dst] "=Tr"(dst.data())
        : "i"(type_traits<typename SubTile::DType>::TypeCode),
          "Tr"(src.data()), "i"(std::remove_reference_t<decltype(src)>::ValidCol),
          "i"(std::remove_reference_t<decltype(src)>::ValidRow),
          "i"(std::remove_reference_t<decltype(src)>::PhysicalCol),
          "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
          "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
          "i"(Opcode), "r"(region_base_units),
          [ElemLayout] "i"(local_layout_code_v<SubTile>)
        : "memory");
  }
}

template <int Opcode, typename Out, typename Parent, typename SubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_scalar(
    Out &dst, region::SubTileView<Parent, SubTile> &src,
    typename SubTile::DType scalar) {
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  volatile typename SubTile::DType value = scalar;
  const uintptr_t region_base_units = src.GetRangeBase();
  {
    static_assert(range::is_legal_subview_parent_v<Parent>,
                  "B.SUBVIEW source must be an assigned Local Matrix Tile with "
                  "a CUBE layout");
    static_assert(SubTile::IsCubeLayout,
                  "B.SUBVIEW source fragment must use a CUBE layout");
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
        "BSTART.TEPL %c10, %D1\n"
        PTO_REGION_ELEMENTWISE_LAYOUT_ASM
        "B.DIM zero, %c3, ->lb0\n"
        "B.DIM zero, %c4, ->lb1\n"
        "B.DIM zero, %c5, ->lb2\n"
        "B.IOT %2, mask=1111, last, ->%0<%Z6>\n"
        "B.SUBVIEW 0, %9, 0, %c7\n"
        "B.IOR [%8],[]\n"
        : [Dst] "=Tr"(dst.data())
        : "i"(type_traits<typename SubTile::DType>::TypeCode),
          "Tr"(src.data()), "i"(std::remove_reference_t<decltype(src)>::ValidCol),
          "i"(std::remove_reference_t<decltype(src)>::ValidRow),
          "i"(SubTile::Cols),
          "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
          "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
          "r"(value), "r"(region_base_units), "i"(Opcode),
          [ElemLayout] "i"(local_layout_code_v<SubTile>)
        : "memory");
  }
}

template <is_tile_data_v Out, typename Parent, typename SubTile>
PTO_REGION_ALWAYS_INLINE void TROWMAX(
    Out &dst, region::SubTileView<Parent, SubTile> &src) {
  pto_region_unary<65>(dst, src);
}

template <is_tile_data_v Out, typename Parent, typename SubTile>
PTO_REGION_ALWAYS_INLINE void TROWSUM(
    Out &dst, region::SubTileView<Parent, SubTile> &src) {
  // TROWSUM derives all three B.DIM operands from the final source view.  In
  // particular, ValidRow is not a lower bound for LB1 and the parent capacity
  // (or an individual assembly writer extent) is never a source dimension.
  static_assert(SubTile::ValidRow > 0 && SubTile::ValidRow <= SubTile::Rows &&
                    SubTile::ValidCol > 0 && SubTile::ValidCol <= SubTile::Cols,
                "TROWSUM source valid shape must fit the source view");
  static_assert(Out::ValidRow == SubTile::ValidRow && Out::ValidCol == 1,
                "TROWSUM destination must be one column with source row count");
  pto_region_unary<64>(dst, src);
}

template <is_tile_data_v Out, typename Parent, typename SubTile>
PTO_REGION_ALWAYS_INLINE void TEXP(
    Out &dst, region::SubTileView<Parent, SubTile> &src) {
  pto_region_unary<18>(dst, src);
}

template <is_tile_data_v Out, typename Parent, typename SubTile>
PTO_REGION_ALWAYS_INLINE void TROWMIN(
    Out &dst, region::SubTileView<Parent, SubTile> &src) {
  pto_region_unary<66>(dst, src);
}

template <is_tile_data_v Out, typename Parent, typename SubTile>
PTO_REGION_ALWAYS_INLINE void TROWPROD(
    Out &dst, region::SubTileView<Parent, SubTile> &src) {
  pto_region_unary<67>(dst, src);
}

template <int ParentSize, bool Init, bool Last, int Opcode,
          typename SubTile, typename In>
PTO_REGION_ALWAYS_INLINE void pto_region_unary_assemble(
    region::TileArrayOutputRef<SubTile> &dst, In &src) {
  static_assert(SubTile::BFractal == BLayout::RowMajor,
                "inline Tile region path requires RowMajor fragments");
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(SubTile::Rows == In::Rows && SubTile::Cols == In::Cols,
                "TileArray slot requires matching physical shape");
  static_assert(SubTile::ValidRow == In::ValidRow &&
                    SubTile::ValidCol == In::ValidCol,
                "TileArray slot requires matching valid shape");
  static_assert(std::is_same_v<typename SubTile::DType, typename In::DType>,
                "TileArray slot requires matching element types");
  // PTO-ISA #265 (issue #702): B.ASSEMBLE field 5 is the WRITER extent in
  // every phase. On INIT the destination B.IOT is the allocating binder and
  // must carry the parent capacity; MIDDLE/LAST reuse the parent register
  // through the destination-only binder (non-allocating, SizeCode=0 form on
  // the parent register) and still encode the nonzero writer extent.
  constexpr int binder_size = Init ? ParentSize : 0;
  constexpr int writer_size =
      tile_type_traits<typename SubTile::TileDType>::TilesizeCode;
  static_assert(writer_size != 0,
                "TileArray slot writer extent must be a nonzero size code");
  // The writer-fits-parent bound is guaranteed by TileArray construction:
  // ParentBytes = slot_count * SubTile::LogicalTileBytes. ParentSize here is
  // only the dispatch case label, and the switch instantiates every case,
  // so a compile-time comparison against it would spuriously fail.
  const uintptr_t range_base_units = dst.range_base_units();
#define PTO_REGION_UNARY_ASSEMBLY_BODY                                      \
  "BSTART.TEPL %c7, %D1\n"                                                \
  "B.DIM zero, %c3, ->lb0\n"                                                    \
  "B.DIM zero, %c4, ->lb1\n"                                                    \
  "B.DIM zero, %c5, ->lb2\n"                                                \
  "B.IOT %2, mask=1111, last, ->%0<%Z6>\n"                                   \
  "B.ASSEMBLE %c10, %c11, %8, 0, %c9\n"
#define PTO_REGION_UNARY_ASSEMBLY_INPUTS                                   \
  "i"(type_traits<typename In::DType>::TypeCode),                          \
  "Tr"(src.data()),                                                       \
  "i"(std::remove_reference_t<decltype(src)>::ValidCol),                  \
  "i"(std::remove_reference_t<decltype(src)>::ValidRow),                  \
  "i"(SubTile::Cols),                                                       \
  "i"(ParentSize),                                                          \
  "i"(Opcode), "r"(range_base_units), "i"(writer_size),                  \
  "i"(Init), "i"(Last)
  if constexpr (Init) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_UNARY_ASSEMBLY_BODY
                 : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                 : PTO_REGION_UNARY_ASSEMBLY_INPUTS
                 : "memory");
  } else {
    // PTO-ISA #265 Local continuation: the final source-form binder slot
    // carries the ParentRef (PTO-B-IOT-STREAM-001: the final effective
    // record has L=1 and earlier records have L=0). The math input shares
    // the final record ahead of the parent register.
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile("BSTART.TEPL %c7, %D1\n"
                 "B.DIM zero, %c3, ->lb0\n"
                 "B.DIM zero, %c4, ->lb1\n"
                 "B.DIM zero, %c5, ->lb2\n"
                 "B.IOT %2, %0, mask=1111, last\n"
                 "B.ASSEMBLE %c10, %c11, %8, 0, %c9\n"
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_UNARY_ASSEMBLY_INPUTS
                 : "memory");
  }
#undef PTO_REGION_UNARY_ASSEMBLY_INPUTS
#undef PTO_REGION_UNARY_ASSEMBLY_BODY
}

template <int ParentSize, int Opcode, typename SubTile, typename In>
PTO_REGION_ALWAYS_INLINE void pto_region_unary_assemble_phase(
    region::TileArrayOutputRef<SubTile> &dst, In &src) {
  if (dst.slot_count() == 1)
    pto_region_unary_assemble<ParentSize, true, true, Opcode>(dst, src);
  else if (dst.ordinal() == 0)
    pto_region_unary_assemble<ParentSize, true, false, Opcode>(dst, src);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_unary_assemble<ParentSize, false, true, Opcode>(dst, src);
  else
    pto_region_unary_assemble<ParentSize, false, false, Opcode>(dst, src);
}

template <int Opcode, typename SubTile, typename In>
PTO_REGION_ALWAYS_INLINE void pto_region_unary_assemble_dispatch(
    region::TileArrayOutputRef<SubTile> &dst, In &src) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_UNARY_PARENT_CASE(N)                                     \
  case N:                                                                   \
    pto_region_unary_assemble_phase<N, Opcode>(dst, src);                    \
    break
    PTO_REGION_UNARY_PARENT_CASE(1);
    PTO_REGION_UNARY_PARENT_CASE(2);
    PTO_REGION_UNARY_PARENT_CASE(3);
    PTO_REGION_UNARY_PARENT_CASE(4);
    PTO_REGION_UNARY_PARENT_CASE(5);
    PTO_REGION_UNARY_PARENT_CASE(6);
    PTO_REGION_UNARY_PARENT_CASE(7);
    PTO_REGION_UNARY_PARENT_CASE(8);
    PTO_REGION_UNARY_PARENT_CASE(9);
    PTO_REGION_UNARY_PARENT_CASE(10);
    PTO_REGION_UNARY_PARENT_CASE(11);
    PTO_REGION_UNARY_PARENT_CASE(12);
#undef PTO_REGION_UNARY_PARENT_CASE
  default:
    __builtin_trap();
  }
}

template <int ParentSize, bool Init, bool Last, int Opcode,
          typename SubTile, typename In>
PTO_REGION_ALWAYS_INLINE void pto_region_scalar_assemble(
    region::TileArrayOutputRef<SubTile> &dst, In &src,
    typename In::DType scalar) {
  static_assert(SubTile::BFractal == BLayout::RowMajor,
                "inline Tile region path requires RowMajor fragments");
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(SubTile::Rows == In::Rows && SubTile::Cols == In::Cols,
                "TileArray slot requires matching physical shape");
  static_assert(std::is_same_v<typename SubTile::DType, typename In::DType>,
                "TileArray slot requires matching element types");
  volatile typename In::DType value = scalar;
  // PTO-ISA #265 (issue #702): field 5 is the writer extent; the INIT
  // destination B.IOT is the allocating binder and carries the parent.
  constexpr int writer_size =
      tile_type_traits<typename SubTile::TileDType>::TilesizeCode;
  static_assert(writer_size != 0,
                "TileArray slot writer extent must be a nonzero size code");
  // The writer-fits-parent bound is guaranteed by TileArray construction:
  // ParentBytes = slot_count * SubTile::LogicalTileBytes. ParentSize here is
  // only the dispatch case label, and the switch instantiates every case,
  // so a compile-time comparison against it would spuriously fail.
  const uintptr_t range_base_units = dst.range_base_units();
#define PTO_REGION_SCALAR_ASSEMBLY_BODY                                    \
  "BSTART.TEPL %c7, %D1\n"                                                \
  "B.DIM zero, %c3, ->lb0\n"                                                  \
  "B.DIM zero, %c4, ->lb1\n"                                                  \
  "B.DIM zero, %c5, ->lb2\n"                                              \
  "B.IOT %2, mask=1111, last, ->%0<%Z6>\n"                                 \
  "B.IOR [%[Scalar]],[]\n"                                                 \
  "B.ASSEMBLE %c10, %c11, %8, 0, %c9\n"
#define PTO_REGION_SCALAR_ASSEMBLY_INPUTS                                  \
  "i"(type_traits<typename In::DType>::TypeCode),                         \
  "Tr"(src.data()),                                                       \
  "i"(std::remove_reference_t<decltype(src)>::ValidCol),                  \
  "i"(std::remove_reference_t<decltype(src)>::ValidRow),                  \
  "i"(SubTile::Cols),                                                       \
  "i"(ParentSize),                                                          \
  "i"(Opcode), "r"(range_base_units), "i"(writer_size),                  \
  "i"(Init), "i"(Last), [Scalar] "r"(value)
  if constexpr (Init) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_SCALAR_ASSEMBLY_BODY
                 : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                 : PTO_REGION_SCALAR_ASSEMBLY_INPUTS
                 : "memory");
  } else {
    // PTO-ISA #265 Local continuation: the final source-form binder slot
    // carries the ParentRef (PTO-B-IOT-STREAM-001: the final effective
    // record has L=1). The math input shares the final record.
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile("BSTART.TEPL %c7, %D1\n"
                 "B.DIM zero, %c3, ->lb0\n"
                 "B.DIM zero, %c4, ->lb1\n"
                 "B.DIM zero, %c5, ->lb2\n"
                 "B.IOT %2, %0, mask=1111, last\n"
                 "B.IOR [%[Scalar]],[]\n"
                 "B.ASSEMBLE %c10, %c11, %8, 0, %c9\n"
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_SCALAR_ASSEMBLY_INPUTS
                 : "memory");
  }
#undef PTO_REGION_SCALAR_ASSEMBLY_INPUTS
#undef PTO_REGION_SCALAR_ASSEMBLY_BODY
}

template <int ParentSize, int Opcode, typename SubTile, typename In>
PTO_REGION_ALWAYS_INLINE void pto_region_scalar_assemble_phase(
    region::TileArrayOutputRef<SubTile> &dst, In &src,
    typename In::DType scalar) {
  if (dst.slot_count() == 1)
    pto_region_scalar_assemble<ParentSize, true, true, Opcode>(dst, src,
                                                                scalar);
  else if (dst.ordinal() == 0)
    pto_region_scalar_assemble<ParentSize, true, false, Opcode>(dst, src,
                                                                 scalar);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_scalar_assemble<ParentSize, false, true, Opcode>(dst, src,
                                                                 scalar);
  else
    pto_region_scalar_assemble<ParentSize, false, false, Opcode>(dst, src,
                                                                  scalar);
}

template <int Opcode, typename SubTile, typename In>
PTO_REGION_ALWAYS_INLINE void pto_region_scalar_assemble_dispatch(
    region::TileArrayOutputRef<SubTile> &dst, In &src,
    typename In::DType scalar) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_SCALAR_PARENT_CASE(N)                                    \
  case N:                                                                   \
    pto_region_scalar_assemble_phase<N, Opcode>(dst, src, scalar);          \
    break
    PTO_REGION_SCALAR_PARENT_CASE(1);
    PTO_REGION_SCALAR_PARENT_CASE(2);
    PTO_REGION_SCALAR_PARENT_CASE(3);
    PTO_REGION_SCALAR_PARENT_CASE(4);
    PTO_REGION_SCALAR_PARENT_CASE(5);
    PTO_REGION_SCALAR_PARENT_CASE(6);
    PTO_REGION_SCALAR_PARENT_CASE(7);
    PTO_REGION_SCALAR_PARENT_CASE(8);
    PTO_REGION_SCALAR_PARENT_CASE(9);
    PTO_REGION_SCALAR_PARENT_CASE(10);
    PTO_REGION_SCALAR_PARENT_CASE(11);
    PTO_REGION_SCALAR_PARENT_CASE(12);
#undef PTO_REGION_SCALAR_PARENT_CASE
  default:
    __builtin_trap();
  }
}

template <int ParentSize, bool Init, bool Last, int Opcode,
          typename SubTile, typename Parent, typename SourceSubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_scalar_subview_assemble(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent, SourceSubTile> &src,
    typename SourceSubTile::DType scalar) {
  static_assert(SubTile::BFractal == BLayout::RowMajor &&
                    SourceSubTile::IsCubeLayout,
                "B.ASSEMBLE destination must be RowMajor and B.SUBVIEW "
                "source must use a CUBE layout");
  static_assert(SubTile::SFractal == SLayout::NoneBox &&
                    SourceSubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(SubTile::Rows == SourceSubTile::Rows &&
                    SubTile::Cols == SourceSubTile::Cols,
                "TileArray slot requires matching physical shape");
  static_assert(SubTile::ValidRow == SourceSubTile::ValidRow &&
                    SubTile::ValidCol == SourceSubTile::ValidCol,
                "TileArray slot requires matching valid shape");
  static_assert(std::is_same_v<typename SubTile::DType,
                               typename SourceSubTile::DType>,
                "TileArray slot requires matching element types");
  volatile typename SourceSubTile::DType value = scalar;
  // PTO-ISA #265 (issue #702): field 5 is the writer extent in every phase.
  constexpr int writer_size =
      tile_type_traits<typename SubTile::TileDType>::TilesizeCode;
  static_assert(writer_size != 0,
                "TileArray slot writer extent must be a nonzero size code");
  // The writer-fits-parent bound is guaranteed by TileArray construction:
  // ParentBytes = slot_count * SubTile::LogicalTileBytes. ParentSize here is
  // only the dispatch case label, and the switch instantiates every case,
  // so a compile-time comparison against it would spuriously fail.
  const uintptr_t source_base_units = src.GetRangeBase();
  const uintptr_t destination_base_units = dst.range_base_units();
#define PTO_REGION_SCALAR_SUBVIEW_ASSEMBLY_BODY                             \
  "BSTART.TEPL %c7, %D1\n"                                                \
  "B.DIM zero, %c3, ->lb0\n"                                                  \
  "B.DIM zero, %c4, ->lb1\n"                                                  \
  "B.DIM zero, %c5, ->lb2\n"                                              \
  "B.IOT %2, mask=1111, last, ->%0<%Z6>\n"                                 \
  "B.SUBVIEW 0, %8, 0, %c9\n"                                             \
  "B.IOR [%14],[]\n"                                                       \
  "B.ASSEMBLE %c12, %c13, %10, 0, %c11\n"
#define PTO_REGION_SCALAR_SUBVIEW_CONT_BODY                                 \
  "BSTART.TEPL %c7, %D1\n"                                                \
  "B.DIM zero, %c3, ->lb0\n"                                                  \
  "B.DIM zero, %c4, ->lb1\n"                                                  \
  "B.DIM zero, %c5, ->lb2\n"                                              \
  "B.IOT %2, %0, mask=1111, last\n"                                       \
  "B.SUBVIEW 0, %8, 0, %c9\n"                                             \
  "B.IOR [%14],[]\n"                                                       \
  "B.ASSEMBLE %c12, %c13, %10, 0, %c11\n"
#define PTO_REGION_SCALAR_SUBVIEW_ASSEMBLY_INPUTS                          \
  "i"(type_traits<typename SourceSubTile::DType>::TypeCode),               \
  "Tr"(src.data()),                                                       \
  "i"(std::remove_reference_t<decltype(src)>::ValidCol),                  \
  "i"(std::remove_reference_t<decltype(src)>::ValidRow),                  \
  "i"(SourceSubTile::Cols),                                                 \
  "i"(ParentSize),                                                          \
  "i"(Opcode), "r"(source_base_units),                                    \
  "i"(tile_type_traits<typename SourceSubTile::TileDType>::TilesizeCode),   \
  "r"(destination_base_units), "i"(writer_size),                          \
  "i"(Init), "i"(Last), "r"(value)
  if constexpr (Init) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_SCALAR_SUBVIEW_ASSEMBLY_BODY
                 : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                 : PTO_REGION_SCALAR_SUBVIEW_ASSEMBLY_INPUTS
                 : "memory");
  } else {
    // PTO-ISA #265 Local continuation: final source-form SizeCode=0 binder.
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_SCALAR_SUBVIEW_CONT_BODY
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_SCALAR_SUBVIEW_ASSEMBLY_INPUTS
                 : "memory");
  }
#undef PTO_REGION_SCALAR_SUBVIEW_ASSEMBLY_INPUTS
#undef PTO_REGION_SCALAR_SUBVIEW_ASSEMBLY_BODY
}

template <int ParentSize, int Opcode, typename SubTile, typename Parent,
          typename SourceSubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_scalar_subview_assemble_phase(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent, SourceSubTile> &src,
    typename SourceSubTile::DType scalar) {
  if (dst.slot_count() == 1)
    pto_region_scalar_subview_assemble<ParentSize, true, true, Opcode>(
        dst, src, scalar);
  else if (dst.ordinal() == 0)
    pto_region_scalar_subview_assemble<ParentSize, true, false, Opcode>(
        dst, src, scalar);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_scalar_subview_assemble<ParentSize, false, true, Opcode>(
        dst, src, scalar);
  else
    pto_region_scalar_subview_assemble<ParentSize, false, false, Opcode>(
        dst, src, scalar);
}

template <int Opcode, typename SubTile, typename Parent,
          typename SourceSubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_scalar_subview_assemble_dispatch(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent, SourceSubTile> &src,
    typename SourceSubTile::DType scalar) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(N)                            \
  case N:                                                                   \
    pto_region_scalar_subview_assemble_phase<N, Opcode>(dst, src, scalar);  \
    break
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(1);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(2);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(3);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(4);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(5);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(6);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(7);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(8);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(9);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(10);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(11);
    PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE(12);
#undef PTO_REGION_SCALAR_SUBVIEW_PARENT_CASE
  default:
    __builtin_trap();
  }
}

template <int ParentSize, bool Init, bool Last, int Opcode,
          typename SubTile, typename Parent, typename SourceSubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_unary_subview_assemble(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent, SourceSubTile> &src) {
  static_assert(SubTile::BFractal == BLayout::RowMajor &&
                    SourceSubTile::IsCubeLayout,
                "B.ASSEMBLE destination must be RowMajor and B.SUBVIEW "
                "source must use a CUBE layout");
  static_assert(SubTile::SFractal == SLayout::NoneBox &&
                    SourceSubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(SubTile::Rows == SourceSubTile::Rows &&
                    SubTile::Cols == SourceSubTile::Cols,
                "TileArray slot requires matching physical shape");
  static_assert(SubTile::ValidRow == SourceSubTile::ValidRow &&
                    SubTile::ValidCol == SourceSubTile::ValidCol,
                "TileArray slot requires matching valid shape");
  static_assert(std::is_same_v<typename SubTile::DType,
                               typename SourceSubTile::DType>,
                "TileArray slot requires matching element types");
  // PTO-ISA #265 (issue #702): field 5 is the writer extent in every phase.
  constexpr int writer_size =
      tile_type_traits<typename SubTile::TileDType>::TilesizeCode;
  static_assert(writer_size != 0,
                "TileArray slot writer extent must be a nonzero size code");
  // The writer-fits-parent bound is guaranteed by TileArray construction:
  // ParentBytes = slot_count * SubTile::LogicalTileBytes. ParentSize here is
  // only the dispatch case label, and the switch instantiates every case,
  // so a compile-time comparison against it would spuriously fail.
  const uintptr_t source_base_units = src.GetRangeBase();
  const uintptr_t destination_base_units = dst.range_base_units();
#define PTO_REGION_UNARY_SUBVIEW_ASSEMBLY_BODY                              \
  "BSTART.TEPL %c7, %D1\n"                                                \
  "B.DIM zero, %c3, ->lb0\n"                                                  \
  "B.DIM zero, %c4, ->lb1\n"                                                  \
  "B.DIM zero, %c5, ->lb2\n"                                              \
  "B.IOT %2, mask=1111, last, ->%0<%Z6>\n"                                 \
  "B.SUBVIEW 0, %8, 0, %c9\n"                                             \
  "B.ASSEMBLE %c12, %c13, %10, 0, %c11\n"
#define PTO_REGION_UNARY_SUBVIEW_CONT_BODY                                  \
  "BSTART.TEPL %c7, %D1\n"                                                \
  "B.DIM zero, %c3, ->lb0\n"                                                  \
  "B.DIM zero, %c4, ->lb1\n"                                                  \
  "B.DIM zero, %c5, ->lb2\n"                                              \
  "B.IOT %2, %0, mask=1111, last\n"                                       \
  "B.SUBVIEW 0, %8, 0, %c9\n"                                             \
  "B.ASSEMBLE %c12, %c13, %10, 0, %c11\n"
#define PTO_REGION_UNARY_SUBVIEW_ASSEMBLY_INPUTS                           \
  "i"(type_traits<typename SourceSubTile::DType>::TypeCode),               \
  "Tr"(src.data()),                                                       \
  "i"(std::remove_reference_t<decltype(src)>::ValidCol),                  \
  "i"(std::remove_reference_t<decltype(src)>::ValidRow),                  \
  "i"(SourceSubTile::Cols),                                                 \
  "i"(ParentSize),                                                          \
  "i"(Opcode), "r"(source_base_units),                                    \
  "i"(tile_type_traits<typename SourceSubTile::TileDType>::TilesizeCode),   \
  "r"(destination_base_units), "i"(writer_size),                          \
  "i"(Init), "i"(Last)
  if constexpr (Init) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_UNARY_SUBVIEW_ASSEMBLY_BODY
                 : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                 : PTO_REGION_UNARY_SUBVIEW_ASSEMBLY_INPUTS
                 : "memory");
  } else {
    // PTO-ISA #265 Local continuation: final source-form SizeCode=0 binder.
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_UNARY_SUBVIEW_CONT_BODY
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_UNARY_SUBVIEW_ASSEMBLY_INPUTS
                 : "memory");
  }
#undef PTO_REGION_UNARY_SUBVIEW_ASSEMBLY_INPUTS
#undef PTO_REGION_UNARY_SUBVIEW_ASSEMBLY_BODY
}

template <int ParentSize, int Opcode, typename SubTile, typename Parent,
          typename SourceSubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_unary_subview_assemble_phase(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent, SourceSubTile> &src) {
  if (dst.slot_count() == 1)
    pto_region_unary_subview_assemble<ParentSize, true, true, Opcode>(dst,
                                                                     src);
  else if (dst.ordinal() == 0)
    pto_region_unary_subview_assemble<ParentSize, true, false, Opcode>(dst,
                                                                      src);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_unary_subview_assemble<ParentSize, false, true, Opcode>(dst,
                                                                      src);
  else
    pto_region_unary_subview_assemble<ParentSize, false, false, Opcode>(dst,
                                                                       src);
}

template <int Opcode, typename SubTile, typename Parent,
          typename SourceSubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_unary_subview_assemble_dispatch(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent, SourceSubTile> &src) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(N)                             \
  case N:                                                                   \
    pto_region_unary_subview_assemble_phase<N, Opcode>(dst, src);           \
    break
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(1);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(2);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(3);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(4);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(5);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(6);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(7);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(8);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(9);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(10);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(11);
    PTO_REGION_UNARY_SUBVIEW_PARENT_CASE(12);
#undef PTO_REGION_UNARY_SUBVIEW_PARENT_CASE
  default:
    __builtin_trap();
  }
}

#define PTO_REGION_UNARY_SOURCE_WRAPPER(Name, Opcode)                        \
  template <is_tile_data_v Out, typename Parent, typename SubTile>           \
  PTO_REGION_ALWAYS_INLINE void Name(                                      \
      Out &dst, region::SubTileView<Parent, SubTile> &src) {                \
    pto_region_unary<Opcode>(dst, src);                                      \
  }

#define PTO_REGION_UNARY_DEST_WRAPPER(Name, Opcode)                          \
  template <typename SubTile, is_tile_data_v In>                             \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      region::TileArrayOutputRef<SubTile> dst, In &src) {                    \
    pto_region_unary_assemble_dispatch<Opcode>(dst, src);                    \
  }

#define PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(Name, Opcode)                  \
  template <typename SubTile, typename Parent, typename SourceSubTile>       \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      region::TileArrayOutputRef<SubTile> dst,                              \
      region::SubTileView<Parent, SourceSubTile> &src) {                    \
    pto_region_unary_subview_assemble_dispatch<Opcode>(dst, src);           \
  }

PTO_REGION_UNARY_SOURCE_WRAPPER(TABS, 15)
PTO_REGION_UNARY_SOURCE_WRAPPER(TNOT, 16)
PTO_REGION_UNARY_SOURCE_WRAPPER(TNEG, 17)
PTO_REGION_UNARY_SOURCE_WRAPPER(TLOG, 19)
PTO_REGION_UNARY_SOURCE_WRAPPER(TRECIP, 20)
PTO_REGION_UNARY_SOURCE_WRAPPER(TSQRT, 21)
PTO_REGION_UNARY_SOURCE_WRAPPER(TRSQRT, 22)
PTO_REGION_UNARY_SOURCE_WRAPPER(TRELU, 23)

#define PTO_REGION_SCALAR_SOURCE_WRAPPER(Name, Opcode)                       \
  template <is_tile_data_v Out, typename Parent, typename SubTile>           \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      Out &dst, region::SubTileView<Parent, SubTile> &src,                   \
      typename SubTile::DType scalar) {                                      \
    pto_region_scalar<Opcode>(dst, src, scalar);                             \
  }

PTO_REGION_SCALAR_SOURCE_WRAPPER(TADDS, 32)
PTO_REGION_SCALAR_SOURCE_WRAPPER(TSUBS, 33)
PTO_REGION_SCALAR_SOURCE_WRAPPER(TMULS, 34)
PTO_REGION_SCALAR_SOURCE_WRAPPER(TDIVS, 35)
PTO_REGION_SCALAR_SOURCE_WRAPPER(TMAXS, 43)
PTO_REGION_SCALAR_SOURCE_WRAPPER(TMINS, 44)

PTO_REGION_UNARY_DEST_WRAPPER(TABS, 15)
PTO_REGION_UNARY_DEST_WRAPPER(TNOT, 16)
PTO_REGION_UNARY_DEST_WRAPPER(TNEG, 17)
PTO_REGION_UNARY_DEST_WRAPPER(TEXP, 18)
PTO_REGION_UNARY_DEST_WRAPPER(TLOG, 19)
PTO_REGION_UNARY_DEST_WRAPPER(TRECIP, 20)
PTO_REGION_UNARY_DEST_WRAPPER(TSQRT, 21)
PTO_REGION_UNARY_DEST_WRAPPER(TRSQRT, 22)
PTO_REGION_UNARY_DEST_WRAPPER(TRELU, 23)

PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(TABS, 15)
PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(TNOT, 16)
PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(TNEG, 17)
PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(TEXP, 18)
PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(TLOG, 19)
PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(TRECIP, 20)
PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(TSQRT, 21)
PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(TRSQRT, 22)
PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER(TRELU, 23)

#define PTO_REGION_SCALAR_DEST_WRAPPER(Name, Opcode)                         \
  template <typename SubTile, is_tile_data_v In>                             \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      region::TileArrayOutputRef<SubTile> dst, In &src,                     \
      typename In::DType scalar) {                                           \
    pto_region_scalar_assemble_dispatch<Opcode>(dst, src, scalar);           \
  }

#define PTO_REGION_SCALAR_SUBVIEW_DEST_WRAPPER(Name, Opcode)                 \
  template <typename SubTile, typename Parent, typename SourceSubTile>       \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      region::TileArrayOutputRef<SubTile> dst,                              \
      region::SubTileView<Parent, SourceSubTile> &src,                      \
      typename SourceSubTile::DType scalar) {                                \
    pto_region_scalar_subview_assemble_dispatch<Opcode>(dst, src, scalar);   \
  }

PTO_REGION_SCALAR_DEST_WRAPPER(TADDS, 32)
PTO_REGION_SCALAR_DEST_WRAPPER(TSUBS, 33)
PTO_REGION_SCALAR_DEST_WRAPPER(TMULS, 34)
PTO_REGION_SCALAR_DEST_WRAPPER(TDIVS, 35)
PTO_REGION_SCALAR_DEST_WRAPPER(TMAXS, 43)
PTO_REGION_SCALAR_DEST_WRAPPER(TMINS, 44)

PTO_REGION_SCALAR_SUBVIEW_DEST_WRAPPER(TADDS, 32)
PTO_REGION_SCALAR_SUBVIEW_DEST_WRAPPER(TSUBS, 33)
PTO_REGION_SCALAR_SUBVIEW_DEST_WRAPPER(TMULS, 34)
PTO_REGION_SCALAR_SUBVIEW_DEST_WRAPPER(TDIVS, 35)
PTO_REGION_SCALAR_SUBVIEW_DEST_WRAPPER(TMAXS, 43)
PTO_REGION_SCALAR_SUBVIEW_DEST_WRAPPER(TMINS, 44)

#undef PTO_REGION_UNARY_SOURCE_WRAPPER
#undef PTO_REGION_UNARY_DEST_WRAPPER
#undef PTO_REGION_UNARY_SUBVIEW_DEST_WRAPPER
#undef PTO_REGION_SCALAR_SOURCE_WRAPPER
#undef PTO_REGION_SCALAR_DEST_WRAPPER
#undef PTO_REGION_SCALAR_SUBVIEW_DEST_WRAPPER

// Raw assembly transport: preserve floating payloads (including NaNs) without
// routing a copy through numeric conversion. The fragment owns the descriptor;
// INIT alone allocates the parent and later slots retain its exact ParentRef.
template <int ParentSize, bool Init, bool Last, typename SubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_tmov_assemble(
    region::TileArrayOutputRef<SubTile> &dst, const SubTile &src) {
  static_assert(is_element_tile_v<SubTile>,
                "TMOV assembly currently requires an ElementTile fragment");
  static_assert(is_element_assembly_fragment_v<SubTile>,
                "TMOV assembly requires a supported full-storage element "
                "fragment");
  const uintptr_t offset = dst.range_base_units();
#define PTO_REGION_TMOV_PREFIX                                               \
  "BSTART.TLSU TMOV, %D[DataType]\n"                                      \
  PTO_TMOV_LAYOUT_ASM                                         \
  "B.DIM zero, %c[ValidCol], ->lb0\n"                                     \
  "B.DIM zero, %c[ValidRow], ->lb1\n"                                     \
  "B.DIM zero, %c[PhysicalCol], ->lb2\n"
#define PTO_REGION_TMOV_ASSEMBLE                                             \
  "B.ASSEMBLE %c[Init], %c[Last], %[Offset], 0, %c[WriterSize]\n"
#define PTO_REGION_TMOV_INPUTS                                               \
  [Src] "Tr"(src.data()),                                                   \
  [DataType] "i"(type_traits<typename SubTile::DType>::TypeCode),             \
  [ValidCol] "i"(SubTile::ValidCol), [ValidRow] "i"(SubTile::ValidRow),        \
  [PhysicalCol] "i"(SubTile::Cols), [ParentSize] "i"(ParentSize),             \
  [WriterSize] "i"(SubTile::TilesizeCode), [Offset] "r"(offset),              \
  [Init] "i"(Init), [Last] "i"(Last),                                        \
  [ElemLayout] "i"(local_layout_code_v<SubTile>)
  if constexpr (Init) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_TMOV_PREFIX
                 "B.IOT %[Src], mask=1111, last, ->%[Dst]<%Z[ParentSize]>\n"
                 PTO_REGION_TMOV_ASSEMBLE
                 : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                 : PTO_REGION_TMOV_INPUTS
                 : "memory");
  } else {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_TMOV_PREFIX
                 "B.IOT %[Src], mask=1111, last\n"
                 "B.IOT %[Dst], mask=1111\n"
                 PTO_REGION_TMOV_ASSEMBLE
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_TMOV_INPUTS
                 : "memory");
  }
#undef PTO_REGION_TMOV_INPUTS
#undef PTO_REGION_TMOV_ASSEMBLE
#undef PTO_REGION_TMOV_PREFIX
}

template <int ParentSize, typename SubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_tmov_phase(
    region::TileArrayOutputRef<SubTile> &dst, const SubTile &src) {
  if (dst.slot_count() == 1)
    pto_region_tmov_assemble<ParentSize, true, true>(dst, src);
  else if (dst.ordinal() == 0)
    pto_region_tmov_assemble<ParentSize, true, false>(dst, src);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_tmov_assemble<ParentSize, false, true>(dst, src);
  else
    pto_region_tmov_assemble<ParentSize, false, false>(dst, src);
}

template <typename SubTile>
  requires(is_element_assembly_fragment_v<SubTile>)
PTO_REGION_ALWAYS_INLINE void TMOV(region::TileArrayOutputRef<SubTile> dst,
                                  const SubTile &src) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_TMOV_PARENT_CASE(N)                                      \
  case N: pto_region_tmov_phase<N>(dst, src); break
    PTO_REGION_TMOV_PARENT_CASE(1);
    PTO_REGION_TMOV_PARENT_CASE(2);
    PTO_REGION_TMOV_PARENT_CASE(3);
    PTO_REGION_TMOV_PARENT_CASE(4);
    PTO_REGION_TMOV_PARENT_CASE(5);
    PTO_REGION_TMOV_PARENT_CASE(6);
    PTO_REGION_TMOV_PARENT_CASE(7);
    PTO_REGION_TMOV_PARENT_CASE(8);
    PTO_REGION_TMOV_PARENT_CASE(9);
    PTO_REGION_TMOV_PARENT_CASE(10);
#undef PTO_REGION_TMOV_PARENT_CASE
  default: __builtin_trap();
  }
}

// B.DATR rounding-mode selector for the TileArray TCVT assembly path. The
// ISA B.DATR RMode field (bits[17:15]) matches the LinxRMode enumeration
// one-to-one; this mirrors the ordinary TCVT_T emitter (jcore/template_asm.hpp)
// so the assembly producer that writes a slot can request MX-scale rounding
// (e.g. LINX_RDN -> RTM floor for BF16 -> E8M0). %D2 is the destination dtype.
#define PTO_REGION_TCVT_DATR_ASM                                             \
  ".if %c[RMode] == 0\nB.DATR %D2, RNONE\n"                                 \
  ".elseif %c[RMode] == 1\nB.DATR %D2, RNE\n"                               \
  ".elseif %c[RMode] == 2\nB.DATR %D2, RTZ\n"                               \
  ".elseif %c[RMode] == 3\nB.DATR %D2, RTM\n"                               \
  ".elseif %c[RMode] == 4\nB.DATR %D2, RTP\n"                               \
  ".elseif %c[RMode] == 5\nB.DATR %D2, RNA\n"                               \
  ".elseif %c[RMode] == 6\nB.DATR %D2, RTO\n"                               \
  ".elseif %c[RMode] == 7\nB.DATR %D2, RHB\n.endif\n"

template <int ParentSize, bool Init, bool Last, int RMode, typename SubTile,
          typename In>
PTO_REGION_ALWAYS_INLINE void
pto_region_tcvt_assemble(region::TileArrayOutputRef<SubTile> &dst, In &src) {
  // TCVT ASL (format-conversion/TCVT.asl): "For a CUBE_M16 or CUBE_M32
  // source, the destination preserves the same CUBE layout and
  // ValidRow/ValidCol, while Row, Col, CELL count, capacity, and packing
  // independently match the destination DataType."  Physical Rows/Cols
  // therefore differ across dtypes by element width; only the logical
  // valid shape and the slot capacity contract are checked here. A
  // cross-dtype physical-Cols equality assert would wrongly reject legal
  // conversions such as BF16 32x2 -> E8M0 32x4 (issue #177).
  static_assert(SubTile::ValidRow == In::ValidRow &&
                    SubTile::ValidCol == In::ValidCol,
                "TCVT assembly slot requires matching valid shape");
  constexpr bool SourceCubeM =
      In::BFractal == BLayout::CubeM16 || In::BFractal == BLayout::CubeM32;
  // TCVT ASL (format-conversion/TCVT.asl) permits CUBE_M16/M32 source and
  // destination geometry, CELL count, capacity, and packing to differ with
  // the destination DataType. Ordinary layouts still preserve physical Row
  // and Col, while each side's capacity follows its own DataType.
  static_assert(SourceCubeM
                    ? (SubTile::BFractal == In::BFractal &&
                       SubTile::IsCubeLayout)
                    : (SubTile::Rows == In::Rows &&
                       SubTile::Cols == In::Cols),
                "TCVT assembly slot requires equal physical Rows/Cols for "
                "ordinary (RowMajor) sources");
  static_assert(RMode >= 0 && RMode <= 7,
                "TCVT RMode must be a LinxRMode value");
  static_assert(
      !(RMode == LINX_RTO &&
        is_tcvt_same_type_round_integral_fp(
            type_traits<typename In::DType>::TypeCode,
            type_traits<typename SubTile::DType>::TypeCode)),
      "TCVT region RTO is illegal for same-type FP16/BF16/FP32/FP64");
  // PTO-ISA #265 (issue #702): field 5 is the writer extent in every phase;
  // the INIT destination B.IOT allocates and carries the parent capacity.
  constexpr int writer_size =
      tile_type_traits<typename SubTile::TileDType>::TilesizeCode;
  static_assert(writer_size != 0,
                "TileArray slot writer extent must be a nonzero size code");
  // The writer-fits-parent bound is guaranteed by TileArray construction:
  // ParentBytes = slot_count * SubTile::LogicalTileBytes. ParentSize here is
  // only the dispatch case label, and the switch instantiates every case,
  // so a compile-time comparison against it would spuriously fail.
  const uintptr_t range_base_units = dst.range_base_units();
#define PTO_REGION_TCVT_ASSEMBLY_PREFIX                                    \
  "BSTART.TEPL 27, %D1\n"                                                  \
  PTO_REGION_TCVT_DATR_ASM                                                 \
  "B.DIM zero, %c4, ->lb0\n"                                               \
  "B.DIM zero, %c5, ->lb1\n"
#define PTO_REGION_TCVT_ASSEMBLY_SUFFIX                                    \
  "B.IOT %3, mask=1111, last, ->%0<%Z7>\n"                                 \
  "B.ASSEMBLE %c10, %c11, %8, 0, %c9\n"
#define PTO_REGION_TCVT_ASSEMBLY_INPUTS                                    \
  "i"(type_traits<typename In::DType>::TypeCode),                           \
  "i"(type_traits<typename SubTile::DType>::TypeCode),                      \
  "Tr"(src.data()),                                                          \
  "i"(std::remove_reference_t<decltype(src)>::ValidCol),                    \
  "i"(std::remove_reference_t<decltype(src)>::ValidRow),                    \
  "i"(SubTile::Cols),                                                        \
  "i"(ParentSize),                                                           \
  "r"(range_base_units), "i"(writer_size), "i"(Init), "i"(Last),            \
  [RMode] "i"(RMode)
  if constexpr (Init) {
    if constexpr (SourceCubeM) {
      /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_TCVT_ASSEMBLY_PREFIX
                   PTO_REGION_TCVT_ASSEMBLY_SUFFIX
                   : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                   : PTO_REGION_TCVT_ASSEMBLY_INPUTS
                   : "memory");
    } else {
      /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_TCVT_ASSEMBLY_PREFIX
                   "B.DIM zero, %c6, ->lb2\n"
                   PTO_REGION_TCVT_ASSEMBLY_SUFFIX
                 : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                 : PTO_REGION_TCVT_ASSEMBLY_INPUTS
                 : "memory");
    }
  } else {
    // PTO-ISA #265 Local continuation: the final source-form binder slot
    // carries the ParentRef (PTO-B-IOT-STREAM-001: the final effective
    // record has L=1). The math input shares the final record.
    if constexpr (SourceCubeM) {
      /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_TCVT_ASSEMBLY_PREFIX
                   "B.IOT %3, %0, mask=1111, last\n"
                   "B.ASSEMBLE %c10, %c11, %8, 0, %c9\n"
                   :
                   : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                     PTO_REGION_TCVT_ASSEMBLY_INPUTS
                   : "memory");
    } else {
      /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_TCVT_ASSEMBLY_PREFIX
                   "B.DIM zero, %c6, ->lb2\n"
                   "B.IOT %3, %0, mask=1111, last\n"
                   "B.ASSEMBLE %c10, %c11, %8, 0, %c9\n"
                   :
                   : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                     PTO_REGION_TCVT_ASSEMBLY_INPUTS
                   : "memory");
    }
  }
#undef PTO_REGION_TCVT_ASSEMBLY_INPUTS
#undef PTO_REGION_TCVT_ASSEMBLY_SUFFIX
#undef PTO_REGION_TCVT_ASSEMBLY_PREFIX
}

template <int ParentSize, int RMode, typename SubTile, typename In>
PTO_REGION_ALWAYS_INLINE void
pto_region_tcvt_phase(region::TileArrayOutputRef<SubTile> &dst, In &src) {
  if (dst.slot_count() == 1)
    pto_region_tcvt_assemble<ParentSize, true, true, RMode>(dst, src);
  else if (dst.ordinal() == 0)
    pto_region_tcvt_assemble<ParentSize, true, false, RMode>(dst, src);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_tcvt_assemble<ParentSize, false, true, RMode>(dst, src);
  else
    pto_region_tcvt_assemble<ParentSize, false, false, RMode>(dst, src);
}

// RMode selects the B.DATR rounding mode (LinxRMode; default LINX_RNONE keeps
// the operation-default encoding). LINX_RDN (RTM floor) is the mode MX E8M0
// scale quantization needs, so the algorithmic producer that writes a slot can
// request it directly, e.g. TCVT<LINX_RDN>(destinations[0][col], scale).
template <int RMode = 0, typename SubTile, is_tile_data_v In>
PTO_REGION_ALWAYS_INLINE void TCVT(region::TileArrayOutputRef<SubTile> dst,
                                   In &src) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_PARENT_CASE(N)                                             \
  case N:                                                                     \
    pto_region_tcvt_phase<N, RMode>(dst, src);                                \
    break
    PTO_REGION_PARENT_CASE(1);
    PTO_REGION_PARENT_CASE(2);
    PTO_REGION_PARENT_CASE(3);
    PTO_REGION_PARENT_CASE(4);
    PTO_REGION_PARENT_CASE(5);
    PTO_REGION_PARENT_CASE(6);
    PTO_REGION_PARENT_CASE(7);
    PTO_REGION_PARENT_CASE(8);
    PTO_REGION_PARENT_CASE(9);
    PTO_REGION_PARENT_CASE(10);
    PTO_REGION_PARENT_CASE(11);
    PTO_REGION_PARENT_CASE(12);
#undef PTO_REGION_PARENT_CASE
  default:
    __builtin_trap();
  }
}
#undef PTO_REGION_TCVT_DATR_ASM

template <int ParentSize, bool Init, bool Last, typename SubTile,
          typename Parent, typename SourceSubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_tcvt_subview_assemble(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent, SourceSubTile> &src) {
  static_assert(SubTile::BFractal == BLayout::RowMajor &&
                    SourceSubTile::IsCubeLayout,
                "B.ASSEMBLE destination must be RowMajor and B.SUBVIEW "
                "source must use a CUBE layout");
  static_assert(SubTile::SFractal == SLayout::NoneBox &&
                    SourceSubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(SubTile::Rows == SourceSubTile::Rows &&
                    SubTile::Cols == SourceSubTile::Cols,
                "TCVT TileArray slot requires matching physical shape");
  static_assert(SubTile::ValidRow == SourceSubTile::ValidRow &&
                    SubTile::ValidCol == SourceSubTile::ValidCol,
                "TCVT TileArray slot requires matching valid shape");
  // PTO-ISA #265 (issue #702): field 5 is the writer extent in every phase.
  constexpr int writer_size =
      tile_type_traits<typename SubTile::TileDType>::TilesizeCode;
  static_assert(writer_size != 0,
                "TileArray slot writer extent must be a nonzero size code");
  // The writer-fits-parent bound is guaranteed by TileArray construction:
  // ParentBytes = slot_count * SubTile::LogicalTileBytes. ParentSize here is
  // only the dispatch case label, and the switch instantiates every case,
  // so a compile-time comparison against it would spuriously fail.
  const uintptr_t source_base_units = src.GetRangeBase();
  const uintptr_t destination_base_units = dst.range_base_units();
#define PTO_REGION_TCVT_SUBVIEW_ASSEMBLY_BODY                               \
  "BSTART.TEPL 27, %D1\n"                                                  \
  "B.DATR %D2, RNONE\n"                                                    \
  "B.DIM zero, %c4, ->lb0\n"                                                   \
  "B.DIM zero, %c5, ->lb1\n"                                                   \
  "B.DIM zero, %c6, ->lb2\n"                                               \
  "B.IOT %3, mask=1111, last, ->%0<%Z7>\n"                                 \
  "B.SUBVIEW 0, %8, 0, %c9\n"                                             \
  "B.ASSEMBLE %c12, %c13, %10, 0, %c11\n"
#define PTO_REGION_TCVT_SUBVIEW_CONT_BODY                                   \
  "BSTART.TEPL 27, %D1\n"                                                  \
  "B.DATR %D2, RNONE\n"                                                    \
  "B.DIM zero, %c4, ->lb0\n"                                                   \
  "B.DIM zero, %c5, ->lb1\n"                                                   \
  "B.DIM zero, %c6, ->lb2\n"                                               \
  "B.IOT %3, %0, mask=1111, last\n"                                       \
  "B.SUBVIEW 0, %8, 0, %c9\n"                                             \
  "B.ASSEMBLE %c12, %c13, %10, 0, %c11\n"
#define PTO_REGION_TCVT_SUBVIEW_ASSEMBLY_INPUTS                            \
  "i"(type_traits<typename SourceSubTile::DType>::TypeCode),               \
  "i"(type_traits<typename SubTile::DType>::TypeCode), "Tr"(src.data()),  \
  "i"(std::remove_reference_t<decltype(src)>::ValidCol),                    \
  "i"(std::remove_reference_t<decltype(src)>::ValidRow),                    \
  "i"(SourceSubTile::Cols),                                                 \
  "i"(ParentSize),                                                           \
  "r"(source_base_units),                                                   \
  "i"(tile_type_traits<typename SourceSubTile::TileDType>::TilesizeCode),   \
  "r"(destination_base_units), "i"(writer_size),                          \
  "i"(Init), "i"(Last)
  if constexpr (Init) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_TCVT_SUBVIEW_ASSEMBLY_BODY
                 : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                 : PTO_REGION_TCVT_SUBVIEW_ASSEMBLY_INPUTS
                 : "memory");
  } else {
    // PTO-ISA #265 Local continuation: final source-form SizeCode=0 binder.
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_TCVT_SUBVIEW_CONT_BODY
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_TCVT_SUBVIEW_ASSEMBLY_INPUTS
                 : "memory");
  }
#undef PTO_REGION_TCVT_SUBVIEW_ASSEMBLY_INPUTS
#undef PTO_REGION_TCVT_SUBVIEW_ASSEMBLY_BODY
}

template <int ParentSize, typename SubTile, typename Parent,
          typename SourceSubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_tcvt_subview_assemble_phase(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent, SourceSubTile> &src) {
  if (dst.slot_count() == 1)
    pto_region_tcvt_subview_assemble<ParentSize, true, true>(dst, src);
  else if (dst.ordinal() == 0)
    pto_region_tcvt_subview_assemble<ParentSize, true, false>(dst, src);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_tcvt_subview_assemble<ParentSize, false, true>(dst, src);
  else
    pto_region_tcvt_subview_assemble<ParentSize, false, false>(dst, src);
}

template <typename SubTile, typename Parent, typename SourceSubTile>
PTO_REGION_ALWAYS_INLINE void TCVT(
    region::TileArrayOutputRef<SubTile> dst,
    region::SubTileView<Parent, SourceSubTile> &src) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(N)                              \
  case N:                                                                   \
    pto_region_tcvt_subview_assemble_phase<N>(dst, src);                    \
    break
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(1);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(2);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(3);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(4);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(5);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(6);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(7);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(8);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(9);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(10);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(11);
    PTO_REGION_TCVT_SUBVIEW_PARENT_CASE(12);
#undef PTO_REGION_TCVT_SUBVIEW_PARENT_CASE
  default:
    __builtin_trap();
  }
}

template <is_tile_data_v Out, typename Parent, typename SubTile>
PTO_REGION_ALWAYS_INLINE void
TCVT(Out &dst, region::SubTileView<Parent, SubTile> &src) {
  constexpr bool source_cube = SubTile::BFractal == BLayout::CubeM16 ||
                               SubTile::BFractal == BLayout::CubeM32;
  static_assert(SubTile::BFractal == BLayout::RowMajor || source_cube,
                "TCVT region source must use RowMajor or CUBE_M16/CUBE_M32");
  static_assert(source_cube ? (Out::BFractal == SubTile::BFractal)
                            : (Out::BFractal == BLayout::RowMajor),
                "TCVT region destination layout is incompatible with source");
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(SubTile::ValidRow == Out::ValidRow &&
                    SubTile::ValidCol == Out::ValidCol,
                "TCVT CUBE_M16/M32 conversion must preserve the valid shape");
  static_assert(is_legal_tcvt_datatype_pair(
                    type_traits<typename SubTile::DType>::TypeCode,
                    type_traits<typename Out::DType>::TypeCode),
                "Illegal TCVT datatype pair");
  const uintptr_t region_base_units = src.GetRangeBase();
  if constexpr (source_cube) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
        "BSTART.TEPL 27, %D1\n"
        "B.DATR %D2, RNONE\n"
        "B.DIM zero, %c4, ->lb0\n"
        "B.DIM zero, %c5, ->lb1\n"
        "B.IOT %3, mask=1111, last, ->%0<%Z6>\n"
        "B.SUBVIEW 0, %7, 0, %c8\n"
        : [Dst] "=Tr"(dst.data())
        : "i"(type_traits<typename SubTile::DType>::TypeCode),
          "i"(type_traits<typename Out::DType>::TypeCode), "Tr"(src.data()),
          "i"(std::remove_reference_t<decltype(src)>::ValidCol),
          "i"(std::remove_reference_t<decltype(src)>::ValidRow),
          "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
          "r"(region_base_units),
          "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode)
        : "memory");
  } else {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
        "BSTART.TEPL 27, %D1\n"
        "B.DATR %D2, RNONE\n"
        "B.DIM zero, %c4, ->lb0\n"
        "B.DIM zero, %c5, ->lb1\n"
        "B.DIM zero, %c6, ->lb2\n"
        "B.IOT %3, mask=1111, last, ->%0<%Z7>\n"
        "B.SUBVIEW 0, %8, 0, %c9\n"
        : [Dst] "=Tr"(dst.data())
        : "i"(type_traits<typename SubTile::DType>::TypeCode),
          "i"(type_traits<typename Out::DType>::TypeCode), "Tr"(src.data()),
          "i"(std::remove_reference_t<decltype(src)>::ValidCol),
          "i"(std::remove_reference_t<decltype(src)>::ValidRow),
          "i"(SubTile::Cols),
          "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
          "r"(region_base_units),
          "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode)
        : "memory");
  }
}

template <int Opcode, typename Out, typename Parent0, typename SubTile0,
          typename Parent1, typename SubTile1>
PTO_REGION_ALWAYS_INLINE void pto_region_binary(
    Out &dst, region::SubTileView<Parent0, SubTile0> &src0,
    region::SubTileView<Parent1, SubTile1> &src1) {
  static_assert((SubTile0::BFractal == BLayout::RowMajor ||
                 SubTile0::IsCubeLayout) &&
                    (SubTile1::BFractal == BLayout::RowMajor ||
                     SubTile1::IsCubeLayout),
                "inline Tile region path requires RowMajor or Cube fragments");
  static_assert(SubTile0::SFractal == SLayout::NoneBox &&
                    SubTile1::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(SubTile0::Rows == SubTile1::Rows &&
                    SubTile0::Cols == SubTile1::Cols,
                "binary region sources require matching physical shapes");
  static_assert(SubTile0::ValidRow == SubTile1::ValidRow &&
                    SubTile0::ValidCol == SubTile1::ValidCol,
                "binary region sources require matching valid shapes");
  static_assert(std::is_same_v<typename SubTile0::DType,
                               typename SubTile1::DType>,
                "binary region sources require matching element types");
  const uintptr_t range_base0_units = src0.GetRangeBase();
  const uintptr_t range_base1_units = src1.GetRangeBase();
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c10, %D1\n"
      "B.DIM zero, %c4, ->lb0\n"
      "B.DIM zero, %c5, ->lb1\n"
      "B.DIM zero, %c6, ->lb2\n"
      "B.IOT %2, %3, mask=1111, last, ->%0<%Z7>\n"
      "B.SUBVIEW 0, %8, 0, %c11\n"
      "B.SUBVIEW 1, %9, 0, %c11\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename SubTile0::DType>::TypeCode),
        "Tr"(src0.data()), "Tr"(src1.data()),
        "i"(std::remove_reference_t<decltype(src0)>::ValidCol),
        "i"(std::remove_reference_t<decltype(src0)>::ValidRow),
        "i"(SubTile0::Cols),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "r"(range_base0_units), "r"(range_base1_units), "i"(Opcode),
        "i"(tile_type_traits<typename SubTile0::TileDType>::TilesizeCode)
      : "memory");
}

template <int Opcode, typename Out, typename Parent, typename SubTile,
          typename Tile>
PTO_REGION_ALWAYS_INLINE void pto_region_binary(
    Out &dst, region::SubTileView<Parent, SubTile> &src0, Tile &src1) {
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(std::is_same_v<typename SubTile::DType, typename Tile::DType>,
                "binary region sources require matching element types");
  static_assert(std::is_same_v<typename SubTile::DType, typename Out::DType>,
                "binary source and destination dtypes must match");
  static_assert(SubTile::Rows == Tile::Rows && SubTile::Cols == Tile::Cols,
                "binary region sources require matching physical shapes");
  static_assert(SubTile::ValidRow == Tile::ValidRow &&
                    SubTile::ValidCol == Tile::ValidCol,
                "binary region sources require matching valid shapes");
  const uintptr_t range_base0_units = src0.GetRangeBase();
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c10, %D1\n"
      "B.DIM zero, %c3, ->lb0\n"
      "B.DIM zero, %c4, ->lb1\n"
      "B.DIM zero, %c5, ->lb2\n"
      "B.IOT %2, %6, mask=1111, last, ->%0<%Z9>\n"
      "B.SUBVIEW 0, %7, 0, %c8\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename SubTile::DType>::TypeCode),
        "Tr"(src0.data()), "i"(Tile::ValidCol), "i"(Tile::ValidRow),
        "i"(Tile::Cols), "Tr"(src1.data()), "r"(range_base0_units),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "i"(Opcode)
      : "memory");
}

template <int Opcode, typename Out, typename Tile, typename Parent,
          typename SubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_binary(
    Out &dst, Tile &src0, region::SubTileView<Parent, SubTile> &src1) {
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(std::is_same_v<typename Tile::DType, typename SubTile::DType>,
                "binary region sources require matching element types");
  static_assert(std::is_same_v<typename Tile::DType, typename Out::DType>,
                "binary source and destination dtypes must match");
  static_assert(Tile::Rows == SubTile::Rows && Tile::Cols == SubTile::Cols,
                "binary region sources require matching physical shapes");
  static_assert(Tile::ValidRow == SubTile::ValidRow &&
                    Tile::ValidCol == SubTile::ValidCol,
                "binary region sources require matching valid shapes");
  const uintptr_t range_base1_units = src1.GetRangeBase();
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c10, %D1\n"
      "B.DIM zero, %c3, ->lb0\n"
      "B.DIM zero, %c4, ->lb1\n"
      "B.DIM zero, %c5, ->lb2\n"
      "B.IOT %6, %2, mask=1111, last, ->%0<%Z9>\n"
      "B.SUBVIEW 1, %7, 0, %c8\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename Tile::DType>::TypeCode), "Tr"(src1.data()),
        "i"(Tile::ValidCol), "i"(Tile::ValidRow), "i"(Tile::Cols),
        "Tr"(src0.data()), "r"(range_base1_units),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "i"(Opcode)
      : "memory");
}

#define PTO_REGION_BINARY_SOURCE_WRAPPER(Name, Opcode)                       \
  template <is_tile_data_v Out, typename Parent0, typename SubTile0,         \
            typename Parent1, typename SubTile1>                              \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      Out &dst, region::SubTileView<Parent0, SubTile0> &src0,                \
      region::SubTileView<Parent1, SubTile1> &src1) {                         \
    pto_region_binary<Opcode>(dst, src0, src1);                              \
  } \
  template <is_tile_data_v Out, typename Parent, typename SubTile,            \
            typename Tile>                                                    \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      Out &dst, region::SubTileView<Parent, SubTile> &src0, Tile &src1) {     \
    pto_region_binary<Opcode>(dst, src0, src1);                              \
  }                                                                          \
  template <is_tile_data_v Out, typename Tile, typename Parent,               \
            typename SubTile>                                                 \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      Out &dst, Tile &src0, region::SubTileView<Parent, SubTile> &src1) {     \
    pto_region_binary<Opcode>(dst, src0, src1);                              \
  }

PTO_REGION_BINARY_SOURCE_WRAPPER(TADD, 0)
PTO_REGION_BINARY_SOURCE_WRAPPER(TSUB, 1)
PTO_REGION_BINARY_SOURCE_WRAPPER(TMUL, 2)
PTO_REGION_BINARY_SOURCE_WRAPPER(TDIV, 3)
PTO_REGION_BINARY_SOURCE_WRAPPER(TREM, 4)
PTO_REGION_BINARY_SOURCE_WRAPPER(TAND, 6)
PTO_REGION_BINARY_SOURCE_WRAPPER(TOR, 7)
PTO_REGION_BINARY_SOURCE_WRAPPER(TXOR, 8)
PTO_REGION_BINARY_SOURCE_WRAPPER(TSHL, 9)
PTO_REGION_BINARY_SOURCE_WRAPPER(TSHR, 10)
PTO_REGION_BINARY_SOURCE_WRAPPER(TMAX, 11)
PTO_REGION_BINARY_SOURCE_WRAPPER(TMIN, 12)

// A row-broadcast source is one logical column (RowMajor or CUBE), or a packed
// CUBE CELL carrier holding up to CubeCellCols logical columns (BF16/FP16 x2 in
// a CUBE_M32 cell, PTO-ISA #207, issue #251).  BroadcastByteOffset selects the
// logical column within the CELL and is bounded separately against CubeCellCols
// above, so the valid-column count is decoupled from the offset.
#define PTO_REGION_ROW_EXPAND_SRC_BCAST(Src)                                   \
  (Src::ValidCol == DYNAMIC || Src::ValidCol == 1 ||                           \
   (Src::IsCubeLayout && Src::ValidCol >= 1 &&                                 \
    Src::ValidCol <= Src::CubeCellCols))

template <int Opcode, typename Out, typename Parent0, typename SubTile0,
          typename Parent1, typename SubTile1, int BroadcastByteOffset = 0>
PTO_REGION_ALWAYS_INLINE void pto_region_expand(
    Out &dst, region::SubTileView<Parent0, SubTile0> &src0,
    region::SubTileView<Parent1, SubTile1> &src1) {
  static_assert(BroadcastByteOffset >= 0 && BroadcastByteOffset <= 7,
                "TROWEXPAND broadcast byte offset must fit B.DATR.RMode");
  static_assert(BroadcastByteOffset == 0 || Out::IsCubeLayout,
                "TROWEXPAND broadcast offset is not valid for RowMajor");
  static_assert(BroadcastByteOffset == 0 ||
                    ((type_traits<typename SubTile1::DType>::bits % 8) == 0 &&
                     BroadcastByteOffset %
                         (type_traits<typename SubTile1::DType>::bits / 8) == 0),
                "TROWEXPAND broadcast byte offset must align to element bytes");
  static_assert(BroadcastByteOffset == 0 ||
                    BroadcastByteOffset /
                            (type_traits<typename SubTile1::DType>::bits / 8) <
                        SubTile1::CubeCellCols,
                "TROWEXPAND broadcast byte offset exceeds CELL columns");
  static_assert(SubTile0::SFractal == SLayout::NoneBox &&
                    SubTile1::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  // src0 is the matrix fragment; only src1 is a row-broadcast carrier
  // (issue #253).
  static_assert(PTO_REGION_ROW_EXPAND_SRC_BCAST(SubTile1),
                "row expansion requires a one-column or packed CUBE CELL "
                "broadcast source (ValidCol <= CubeCellCols)");
  static_assert(SubTile0::ValidCol == Out::ValidCol,
                "row expansion matrix source cols must match the destination");
  static_assert(SubTile0::ValidRow == SubTile1::ValidRow,
                "row expansion broadcast rows must match the matrix");
  static_assert(std::is_same_v<typename SubTile0::DType,
                               typename SubTile1::DType>,
                "expansion sources require matching element types");
  static_assert(std::is_same_v<typename SubTile0::DType, typename Out::DType>,
                "expansion source and destination dtypes must match");
  static_assert(Out::ValidCol > 0 && Out::ValidRow > 0,
                "region expansion requires a static destination shape");
  const uintptr_t range_base0_units = src0.GetRangeBase();
  const uintptr_t range_base1_units = src1.GetRangeBase();
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c10, %D1\n"
      "B.DIM zero, %c4, ->lb0\n"
      "B.DIM zero, %c5, ->lb1\n"
      "B.DIM zero, %c6, ->lb2\n"
      "B.IOT %2, %3, mask=1111, last, ->%0<%Z7>\n"
      "B.SUBVIEW 0, %8, 0, %c11\n"
      "B.SUBVIEW 1, %9, 0, %c14\n"
      ".if %c12 == 29\n"
      ".if %c13 == 0\n"
      "B.DATR CUBE_M32, Null\n"
      ".else\n"
      "B.DATR CUBE_M32, DTYPE_NONE, Null, EQ, %c13\n"
      ".endif\n"
      ".elseif %c12 == 31\n"
      ".if %c13 == 0\n"
      "B.DATR CUBE_M16, Null\n"
      ".else\n"
      "B.DATR CUBE_M16, DTYPE_NONE, Null, EQ, %c13\n"
      ".endif\n"
      ".endif\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename SubTile0::DType>::TypeCode),
        "Tr"(src0.data()), "Tr"(src1.data()), "i"(Out::ValidCol),
        "i"(Out::ValidRow), "i"(Out::Cols),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "r"(range_base0_units), "r"(range_base1_units), "i"(Opcode),
        "i"(tile_type_traits<typename SubTile0::TileDType>::TilesizeCode),
        "i"(local_layout_code_v<Out>), "i"(BroadcastByteOffset),
        "i"(tile_type_traits<typename SubTile1::TileDType>::TilesizeCode)
      : "memory");
}

template <int Opcode, typename Out, typename Parent, typename SubTile,
          typename Tile, int BroadcastByteOffset = 0>
PTO_REGION_ALWAYS_INLINE void pto_region_row_expand(
    Out &dst, region::SubTileView<Parent, SubTile> &src0, Tile &src1) {
  static_assert(BroadcastByteOffset >= 0 && BroadcastByteOffset <= 7,
                "TROWEXPAND broadcast byte offset must fit B.DATR.RMode");
  static_assert(BroadcastByteOffset == 0 || Out::IsCubeLayout,
                "TROWEXPAND broadcast offset is not valid for RowMajor");
  static_assert(BroadcastByteOffset == 0 ||
                    ((type_traits<typename SubTile::DType>::bits % 8) == 0 &&
                     BroadcastByteOffset %
                         (type_traits<typename SubTile::DType>::bits / 8) == 0),
                "TROWEXPAND broadcast byte offset must align to element bytes");
  static_assert(BroadcastByteOffset == 0 ||
                    BroadcastByteOffset /
                            (type_traits<typename SubTile::DType>::bits / 8) <
                        SubTile::CubeCellCols,
                "TROWEXPAND broadcast byte offset exceeds CELL columns");
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(std::is_same_v<typename SubTile::DType, typename Tile::DType>,
                "expansion sources require matching element types");
  static_assert(std::is_same_v<typename SubTile::DType, typename Out::DType>,
                "expansion source and destination dtypes must match");
  static_assert(PTO_REGION_ROW_EXPAND_SRC_BCAST(Tile),
                "row expansion requires a one-column or packed CUBE CELL "
                "broadcast source (ValidCol <= CubeCellCols)");
  static_assert(SubTile::ValidRow == Tile::ValidRow,
                "row expansion broadcast rows must match the matrix");
  static_assert(Out::ValidCol > 0 && Out::ValidRow > 0,
                "region expansion requires a static destination shape");
  const uintptr_t range_base0_units = src0.GetRangeBase();
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c10, %D1\n"
      "B.DIM zero, %c3, ->lb0\n"
      "B.DIM zero, %c4, ->lb1\n"
      "B.DIM zero, %c5, ->lb2\n"
      "B.IOT %2, %6, mask=1111, last, ->%0<%Z9>\n"
      "B.SUBVIEW 0, %7, 0, %c8\n"
      ".if %c11 == 29\n"
      ".if %c12 == 0\n"
      "B.DATR CUBE_M32, Null\n"
      ".else\n"
      "B.DATR CUBE_M32, DTYPE_NONE, Null, EQ, %c12\n"
      ".endif\n"
      ".elseif %c11 == 31\n"
      ".if %c12 == 0\n"
      "B.DATR CUBE_M16, Null\n"
      ".else\n"
      "B.DATR CUBE_M16, DTYPE_NONE, Null, EQ, %c12\n"
      ".endif\n"
      ".endif\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename SubTile::DType>::TypeCode),
        "Tr"(src0.data()), "i"(Out::ValidCol), "i"(Out::ValidRow),
        "i"(Out::Cols), "Tr"(src1.data()),
        "r"(range_base0_units),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "i"(Opcode), "i"(local_layout_code_v<Out>), "i"(BroadcastByteOffset)
      : "memory");
}

template <int Opcode, typename Out, typename Tile, typename Parent,
          typename SubTile, int BroadcastByteOffset = 0>
PTO_REGION_ALWAYS_INLINE void pto_region_row_expand(
    Out &dst, Tile &src0, region::SubTileView<Parent, SubTile> &src1) {
  static_assert(BroadcastByteOffset >= 0 && BroadcastByteOffset <= 7,
                "TROWEXPAND broadcast byte offset must fit B.DATR.RMode");
  static_assert(BroadcastByteOffset == 0 || Out::IsCubeLayout,
                "TROWEXPAND broadcast offset is not valid for RowMajor");
  static_assert(BroadcastByteOffset == 0 ||
                    ((type_traits<typename SubTile::DType>::bits % 8) == 0 &&
                     BroadcastByteOffset %
                         (type_traits<typename SubTile::DType>::bits / 8) == 0),
                "TROWEXPAND broadcast byte offset must align to element bytes");
  static_assert(BroadcastByteOffset == 0 ||
                    BroadcastByteOffset /
                            (type_traits<typename SubTile::DType>::bits / 8) <
                        SubTile::CubeCellCols,
                "TROWEXPAND broadcast byte offset exceeds CELL columns");
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(std::is_same_v<typename Tile::DType, typename SubTile::DType>,
                "expansion sources require matching element types");
  static_assert(std::is_same_v<typename Tile::DType, typename Out::DType>,
                "expansion source and destination dtypes must match");
  static_assert(PTO_REGION_ROW_EXPAND_SRC_BCAST(SubTile),
                "row expansion requires a one-column or packed CUBE CELL "
                "broadcast source (ValidCol <= CubeCellCols)");
  static_assert(Tile::ValidRow == SubTile::ValidRow,
                "row expansion broadcast rows must match the matrix");
  static_assert(Out::ValidCol > 0 && Out::ValidRow > 0,
                "region expansion requires a static destination shape");
  const uintptr_t range_base1_units = src1.GetRangeBase();
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c10, %D1\n"
      "B.DIM zero, %c3, ->lb0\n"
      "B.DIM zero, %c4, ->lb1\n"
      "B.DIM zero, %c5, ->lb2\n"
      "B.IOT %6, %2, mask=1111, last, ->%0<%Z9>\n"
      "B.SUBVIEW 1, %7, 0, %c8\n"
      ".if %c11 == 29\n"
      ".if %c12 == 0\n"
      "B.DATR CUBE_M32, Null\n"
      ".else\n"
      "B.DATR CUBE_M32, DTYPE_NONE, Null, EQ, %c12\n"
      ".endif\n"
      ".elseif %c11 == 31\n"
      ".if %c12 == 0\n"
      "B.DATR CUBE_M16, Null\n"
      ".else\n"
      "B.DATR CUBE_M16, DTYPE_NONE, Null, EQ, %c12\n"
      ".endif\n"
      ".endif\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename Tile::DType>::TypeCode), "Tr"(src1.data()),
        "i"(Out::ValidCol), "i"(Out::ValidRow), "i"(Out::Cols),
        "Tr"(src0.data()), "r"(range_base1_units),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "i"(Opcode), "i"(local_layout_code_v<Out>), "i"(BroadcastByteOffset)
      : "memory");
}

template <int Opcode, typename Out, typename Parent, typename SubTile,
          typename Tile>
PTO_REGION_ALWAYS_INLINE void pto_region_col_expand(
    Out &dst, region::SubTileView<Parent, SubTile> &src0, Tile &src1) {
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(std::is_same_v<typename SubTile::DType, typename Tile::DType>,
                "expansion sources require matching element types");
  static_assert(std::is_same_v<typename SubTile::DType, typename Out::DType>,
                "expansion source and destination dtypes must match");
  static_assert(Tile::ValidRow == 1,
                "column expansion requires a one-row broadcast source");
  static_assert(SubTile::ValidCol == Tile::ValidCol,
                "column expansion broadcast cols must match the matrix");
  static_assert(Out::ValidCol > 0 && Out::ValidRow > 0,
                "region expansion requires a static destination shape");
  static_assert(Out::BFractal == SubTile::BFractal &&
                    Out::BFractal == Tile::BFractal,
                "column expansion sources must match the destination layout");
  const uintptr_t range_base0_units = src0.GetRangeBase();
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c10, %D1\n"
      PTO_REGION_ELEMENTWISE_LAYOUT_ASM
      "B.DIM zero, %c3, ->lb0\n"
      "B.DIM zero, %c4, ->lb1\n"
      "B.DIM zero, %c5, ->lb2\n"
      "B.IOT %2, %6, mask=1111, last, ->%0<%Z9>\n"
      "B.SUBVIEW 0, %7, 0, %c8\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename SubTile::DType>::TypeCode),
        "Tr"(src0.data()), "i"(Out::ValidCol), "i"(Out::ValidRow),
        "i"(Out::Cols), "Tr"(src1.data()),
        "r"(range_base0_units),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "i"(Opcode), [ElemLayout] "i"(local_layout_code_v<Out>)
      : "memory");
}

template <int Opcode, typename Out, typename Tile, typename Parent,
          typename SubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_col_expand(
    Out &dst, Tile &src0, region::SubTileView<Parent, SubTile> &src1) {
  static_assert(SubTile::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(std::is_same_v<typename Tile::DType, typename SubTile::DType>,
                "expansion sources require matching element types");
  static_assert(std::is_same_v<typename Tile::DType, typename Out::DType>,
                "expansion source and destination dtypes must match");
  static_assert(SubTile::ValidRow == 1,
                "column expansion requires a one-row broadcast source");
  static_assert(Tile::ValidCol == SubTile::ValidCol,
                "column expansion broadcast cols must match the matrix");
  static_assert(Out::ValidCol > 0 && Out::ValidRow > 0,
                "region expansion requires a static destination shape");
  static_assert(Out::BFractal == SubTile::BFractal &&
                    Out::BFractal == Tile::BFractal,
                "column expansion sources must match the destination layout");
  const uintptr_t range_base1_units = src1.GetRangeBase();
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c10, %D1\n"
      PTO_REGION_ELEMENTWISE_LAYOUT_ASM
      "B.DIM zero, %c3, ->lb0\n"
      "B.DIM zero, %c4, ->lb1\n"
      "B.DIM zero, %c5, ->lb2\n"
      "B.IOT %6, %2, mask=1111, last, ->%0<%Z9>\n"
      "B.SUBVIEW 1, %7, 0, %c8\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename Tile::DType>::TypeCode), "Tr"(src1.data()),
        "i"(Out::ValidCol), "i"(Out::ValidRow), "i"(Out::Cols),
        "Tr"(src0.data()), "r"(range_base1_units),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "i"(Opcode), [ElemLayout] "i"(local_layout_code_v<Out>)
      : "memory");
}

#undef PTO_REGION_ROW_EXPAND_SRC_BCAST

#define PTO_REGION_EXPAND_SOURCE_WRAPPER(Name, Opcode)                       \
  template <is_tile_data_v Out, typename Parent0, typename SubTile0,          \
            typename Parent1, typename SubTile1, int BroadcastByteOffset = 0> \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      Out &dst, region::SubTileView<Parent0, SubTile0> &src0,                \
      region::SubTileView<Parent1, SubTile1> &src1) {                         \
    pto_region_expand<Opcode, Out, Parent0, SubTile0, Parent1, SubTile1,       \
                      BroadcastByteOffset>(dst, src0, src1);                 \
  }

#define PTO_REGION_ROW_EXPAND_MIXED_WRAPPER(Name, Opcode)                    \
  template <is_tile_data_v Out, typename Parent, typename SubTile,            \
            typename Tile, int BroadcastByteOffset = 0>                       \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      Out &dst, region::SubTileView<Parent, SubTile> &src0, Tile &src1) {     \
    pto_region_row_expand<Opcode, Out, Parent, SubTile, Tile,                 \
                          BroadcastByteOffset>(dst, src0, src1);              \
  }                                                                          \
  template <is_tile_data_v Out, typename Tile, typename Parent,               \
            typename SubTile, int BroadcastByteOffset = 0>                    \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      Out &dst, Tile &src0, region::SubTileView<Parent, SubTile> &src1) {     \
    pto_region_row_expand<Opcode, Out, Tile, Parent, SubTile,                 \
                          BroadcastByteOffset>(dst, src0, src1);              \
  }

#define PTO_REGION_COL_EXPAND_MIXED_WRAPPER(Name, Opcode)                    \
  template <is_tile_data_v Out, typename Parent, typename SubTile,            \
            typename Tile>                                                    \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      Out &dst, region::SubTileView<Parent, SubTile> &src0, Tile &src1) {     \
    pto_region_col_expand<Opcode>(dst, src0, src1);                          \
  }                                                                          \
  template <is_tile_data_v Out, typename Tile, typename Parent,               \
            typename SubTile>                                                 \
  PTO_REGION_ALWAYS_INLINE void Name(                                       \
      Out &dst, Tile &src0, region::SubTileView<Parent, SubTile> &src1) {     \
    pto_region_col_expand<Opcode>(dst, src0, src1);                          \
  }

PTO_REGION_EXPAND_SOURCE_WRAPPER(TROWEXPANDADD, 69)
PTO_REGION_EXPAND_SOURCE_WRAPPER(TROWEXPANDSUB, 70)
PTO_REGION_EXPAND_SOURCE_WRAPPER(TROWEXPANDMUL, 71)
PTO_REGION_EXPAND_SOURCE_WRAPPER(TROWEXPANDDIV, 72)
PTO_REGION_EXPAND_SOURCE_WRAPPER(TROWEXPANDMAX, 73)
PTO_REGION_EXPAND_SOURCE_WRAPPER(TROWEXPANDMIN, 74)
PTO_REGION_EXPAND_SOURCE_WRAPPER(TROWEXPANDEXPDIF, 75)
PTO_REGION_ROW_EXPAND_MIXED_WRAPPER(TROWEXPANDADD, 69)
PTO_REGION_ROW_EXPAND_MIXED_WRAPPER(TROWEXPANDSUB, 70)
PTO_REGION_ROW_EXPAND_MIXED_WRAPPER(TROWEXPANDMUL, 71)
PTO_REGION_ROW_EXPAND_MIXED_WRAPPER(TROWEXPANDDIV, 72)
PTO_REGION_ROW_EXPAND_MIXED_WRAPPER(TROWEXPANDMAX, 73)
PTO_REGION_ROW_EXPAND_MIXED_WRAPPER(TROWEXPANDMIN, 74)
PTO_REGION_ROW_EXPAND_MIXED_WRAPPER(TROWEXPANDEXPDIF, 75)
PTO_REGION_COL_EXPAND_MIXED_WRAPPER(TCOLEXPANDADD, 85)
PTO_REGION_COL_EXPAND_MIXED_WRAPPER(TCOLEXPANDSUB, 86)
PTO_REGION_COL_EXPAND_MIXED_WRAPPER(TCOLEXPANDMUL, 87)
PTO_REGION_COL_EXPAND_MIXED_WRAPPER(TCOLEXPANDDIV, 88)
PTO_REGION_COL_EXPAND_MIXED_WRAPPER(TCOLEXPANDMAX, 89)
PTO_REGION_COL_EXPAND_MIXED_WRAPPER(TCOLEXPANDMIN, 90)
PTO_REGION_COL_EXPAND_MIXED_WRAPPER(TCOLEXPANDEXPDIF, 91)

#undef PTO_REGION_EXPAND_SOURCE_WRAPPER
#undef PTO_REGION_ROW_EXPAND_MIXED_WRAPPER
#undef PTO_REGION_COL_EXPAND_MIXED_WRAPPER

template <int Opcode, typename Out, typename Tile, typename Parent,
          typename SubTile>
PTO_REGION_ALWAYS_INLINE void pto_region_binary_reduction_prefix(
    Out &dst, Tile &src0,
    region::ReductionPrefixView<Parent, SubTile> &src1) {
  static_assert(Tile::IsCubeLayout && SubTile::IsCubeLayout,
                "reduction prefix sources require CUBE layouts");
  static_assert(Tile::Rows == SubTile::Rows && Tile::Cols == SubTile::Cols,
                "reduction prefix sources require matching physical shapes");
  static_assert(Tile::ValidRow == SubTile::ValidRow &&
                    Tile::ValidCol == SubTile::ValidCol,
                "reduction prefix sources require matching valid shapes");
  static_assert(std::is_same_v<typename Tile::DType, typename SubTile::DType>,
                "reduction prefix sources require matching element types");
  static_assert(Tile::BFractal == SubTile::BFractal,
                "reduction prefix sources require matching CUBE layouts");
  const uintptr_t prefix_base_units = src1.GetRangeBase();
  // A dynamic (ValidRow == -1) tile carries its valid row count at runtime;
  // B.DIM's uimm17 is unsigned, so the count must travel in a GPR via the
  // "B.DIM RegSrc, uimm17, ->LBn" form (issue #187), exactly like the TCVT
  // dynamic branch in template_asm.hpp.
  if constexpr (Tile::ValidRow < 0) {
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c8, %D1\n"
      PTO_REGION_ELEMENTWISE_LAYOUT_ASM
      "B.DIM zero, %c4, ->lb0\n"
      "B.DIM %[DynValidRow], 0, ->lb1\n"
      "B.DIM zero, %c5, ->lb2\n"
      "B.IOT %2, %3, mask=1111, last, ->%0<%Z6>\n"
      "B.SUBVIEW 1, %7, 0, %c9\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename Tile::DType>::TypeCode),
        "Tr"(src0.data()), "Tr"(src1.data()),
        "i"(Tile::ValidCol), "i"(Tile::Cols),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "r"(prefix_base_units), "i"(Opcode),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        [ElemLayout] "i"(local_layout_code_v<Tile>),
        [DynValidRow] "r"(src0.GetValidRow())
      : "memory");
  } else {
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c9, %D1\n"
      PTO_REGION_ELEMENTWISE_LAYOUT_ASM
      "B.DIM zero, %c4, ->lb0\n"
      "B.DIM zero, %c5, ->lb1\n"
      "B.DIM zero, %c6, ->lb2\n"
      "B.IOT %2, %3, mask=1111, last, ->%0<%Z7>\n"
      "B.SUBVIEW 1, %8, 0, %c10\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename Tile::DType>::TypeCode),
        "Tr"(src0.data()), "Tr"(src1.data()),
        "i"(Tile::ValidCol), "i"(Tile::ValidRow), "i"(Tile::Cols),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "r"(prefix_base_units), "i"(Opcode),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        [ElemLayout] "i"(local_layout_code_v<Tile>)
      : "memory");
  }
}

template <int Opcode, typename Out, typename Parent, typename SubTile,
          typename Tile>
PTO_REGION_ALWAYS_INLINE void pto_region_binary_reduction_prefix(
    Out &dst, region::ReductionPrefixView<Parent, SubTile> &src0,
    Tile &src1) {
  static_assert(Tile::IsCubeLayout && SubTile::IsCubeLayout,
                "reduction prefix sources require CUBE layouts");
  static_assert(Tile::Rows == SubTile::Rows && Tile::Cols == SubTile::Cols,
                "reduction prefix sources require matching physical shapes");
  static_assert(Tile::ValidRow == SubTile::ValidRow &&
                    Tile::ValidCol == SubTile::ValidCol,
                "reduction prefix sources require matching valid shapes");
  static_assert(std::is_same_v<typename Tile::DType, typename SubTile::DType>,
                "reduction prefix sources require matching element types");
  static_assert(Tile::BFractal == SubTile::BFractal,
                "reduction prefix sources require matching CUBE layouts");
  const uintptr_t prefix_base_units = src0.GetRangeBase();
  // Dynamic ValidRow travels in a GPR (issue #187, see the mirrored
  // Tile-src overload above).
  if constexpr (SubTile::ValidRow < 0) {
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c8, %D1\n"
      PTO_REGION_ELEMENTWISE_LAYOUT_ASM
      "B.DIM zero, %c4, ->lb0\n"
      "B.DIM %[DynValidRow], 0, ->lb1\n"
      "B.DIM zero, %c5, ->lb2\n"
      "B.IOT %2, %3, mask=1111, last, ->%0<%Z6>\n"
      "B.SUBVIEW 0, %7, 0, %c9\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename SubTile::DType>::TypeCode),
        "Tr"(src0.data()), "Tr"(src1.data()),
        "i"(SubTile::ValidCol), "i"(SubTile::Cols),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "r"(prefix_base_units), "i"(Opcode),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        [ElemLayout] "i"(local_layout_code_v<SubTile>),
        [DynValidRow] "r"(src1.GetValidRow())
      : "memory");
  } else {
  /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
      "BSTART.TEPL %c9, %D1\n"
      PTO_REGION_ELEMENTWISE_LAYOUT_ASM
      "B.DIM zero, %c4, ->lb0\n"
      "B.DIM zero, %c5, ->lb1\n"
      "B.DIM zero, %c6, ->lb2\n"
      "B.IOT %2, %3, mask=1111, last, ->%0<%Z7>\n"
      "B.SUBVIEW 0, %8, 0, %c10\n"
      : [Dst] "=Tr"(dst.data())
      : "i"(type_traits<typename SubTile::DType>::TypeCode),
        "Tr"(src0.data()), "Tr"(src1.data()),
        "i"(SubTile::ValidCol), "i"(SubTile::ValidRow), "i"(SubTile::Cols),
        "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
        "r"(prefix_base_units), "i"(Opcode),
        "i"(tile_type_traits<typename SubTile::TileDType>::TilesizeCode),
        [ElemLayout] "i"(local_layout_code_v<SubTile>)
      : "memory");
  }
}

template <int Opcode, typename Out, typename Parent0, typename SubTile0,
          typename Parent1, typename SubTile1>
PTO_REGION_ALWAYS_INLINE void pto_region_binary_reduction_prefix(
    Out &dst,
    region::ReductionPrefixView<Parent0, SubTile0> &src0,
    region::ReductionPrefixView<Parent1, SubTile1> &src1) {
  static_assert(SubTile0::IsCubeLayout && SubTile1::IsCubeLayout,
                "reduction prefix sources require CUBE layouts");
  static_assert(SubTile0::Rows == SubTile1::Rows &&
                    SubTile0::Cols == SubTile1::Cols,
                "reduction prefix sources require matching physical shapes");
  static_assert(SubTile0::ValidRow == SubTile1::ValidRow &&
                    SubTile0::ValidCol == SubTile1::ValidCol,
                "reduction prefix sources require matching valid shapes");
  static_assert(std::is_same_v<typename SubTile0::DType,
                               typename SubTile1::DType>,
                "reduction prefix sources require matching element types");
  static_assert(SubTile0::BFractal == SubTile1::BFractal,
                "reduction prefix sources require matching CUBE layouts");

  const uintptr_t prefix_base0_units = src0.GetRangeBase();
  const uintptr_t prefix_base1_units = src1.GetRangeBase();
  // ReductionPrefixView carries a statically selected CELL. Keep the dynamic
  // branch in lockstep with the Tile/prefix overloads so that a dynamic prefix
  // remains well-formed and uses the register B.DIM form.
  if constexpr (SubTile0::ValidRow < 0) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
        "BSTART.TEPL %c9, %D1\n"
        PTO_REGION_ELEMENTWISE_LAYOUT_ASM
        "B.DIM zero, %c4, ->lb0\n"
        "B.DIM %[DynValidRow], 0, ->lb1\n"
        "B.DIM zero, %c5, ->lb2\n"
        "B.IOT %2, %3, mask=1111, last, ->%0<%Z6>\n"
        "B.SUBVIEW 0, %7, 0, %c10\n"
        "B.SUBVIEW 1, %8, 0, %c10\n"
        : [Dst] "=Tr"(dst.data())
        : "i"(type_traits<typename SubTile0::DType>::TypeCode),
          "Tr"(src0.data()), "Tr"(src1.data()),
          "i"(SubTile0::ValidCol), "i"(SubTile0::Cols),
          "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
          "r"(prefix_base0_units), "r"(prefix_base1_units), "i"(Opcode),
          "i"(tile_type_traits<typename SubTile0::TileDType>::TilesizeCode),
          [ElemLayout] "i"(local_layout_code_v<SubTile0>),
          [DynValidRow] "r"(src0.GetValidRow())
        : "memory");
  } else {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
        "BSTART.TEPL %c10, %D1\n"
        PTO_REGION_ELEMENTWISE_LAYOUT_ASM
        "B.DIM zero, %c4, ->lb0\n"
        "B.DIM zero, %c5, ->lb1\n"
        "B.DIM zero, %c6, ->lb2\n"
        "B.IOT %2, %3, mask=1111, last, ->%0<%Z7>\n"
        "B.SUBVIEW 0, %8, 0, %c11\n"
        "B.SUBVIEW 1, %9, 0, %c11\n"
        : [Dst] "=Tr"(dst.data())
        : "i"(type_traits<typename SubTile0::DType>::TypeCode),
          "Tr"(src0.data()), "Tr"(src1.data()),
          "i"(SubTile0::ValidCol), "i"(SubTile0::ValidRow),
          "i"(SubTile0::Cols),
          "i"(tile_type_traits<typename Out::TileDType>::TilesizeCode),
          "r"(prefix_base0_units), "r"(prefix_base1_units), "i"(Opcode),
          "i"(tile_type_traits<typename SubTile0::TileDType>::TilesizeCode),
          [ElemLayout] "i"(local_layout_code_v<SubTile0>)
        : "memory");
  }
}

#define PTO_REGION_BINARY_PREFIX_WRAPPER(Name, Opcode)                         \
  template <is_tile_data_v Out, is_tile_data_v Tile, typename Parent, typename SubTile> \
  PTO_REGION_ALWAYS_INLINE void Name(                                        \
      Out &dst, Tile &src0,                                                   \
      region::ReductionPrefixView<Parent, SubTile> &src1) {                    \
    pto_region_binary_reduction_prefix<Opcode>(dst, src0, src1);              \
  }                                                                            \
  template <is_tile_data_v Out, typename Parent, typename SubTile,             \
            is_tile_data_v Tile>                                               \
  PTO_REGION_ALWAYS_INLINE void Name(                                        \
      Out &dst, region::ReductionPrefixView<Parent, SubTile> &src0,           \
      Tile &src1) {                                                           \
    pto_region_binary_reduction_prefix<Opcode>(dst, src0, src1);              \
  }                                                                            \
  template <is_tile_data_v Out, typename Parent0, typename SubTile0,           \
            typename Parent1, typename SubTile1>                               \
  PTO_REGION_ALWAYS_INLINE void Name(                                        \
      Out &dst,                                                               \
      region::ReductionPrefixView<Parent0, SubTile0> &src0,                   \
      region::ReductionPrefixView<Parent1, SubTile1> &src1) {                 \
    pto_region_binary_reduction_prefix<Opcode>(dst, src0, src1);              \
  }

PTO_REGION_BINARY_PREFIX_WRAPPER(TADD, 0)
PTO_REGION_BINARY_PREFIX_WRAPPER(TSUB, 1)
PTO_REGION_BINARY_PREFIX_WRAPPER(TMUL, 2)
PTO_REGION_BINARY_PREFIX_WRAPPER(TDIV, 3)
PTO_REGION_BINARY_PREFIX_WRAPPER(TREM, 4)
PTO_REGION_BINARY_PREFIX_WRAPPER(TAND, 6)
PTO_REGION_BINARY_PREFIX_WRAPPER(TOR, 7)
PTO_REGION_BINARY_PREFIX_WRAPPER(TXOR, 8)
PTO_REGION_BINARY_PREFIX_WRAPPER(TMAX, 11)
PTO_REGION_BINARY_PREFIX_WRAPPER(TMIN, 12)

#undef PTO_REGION_BINARY_PREFIX_WRAPPER

template <int ParentSize, bool Init, bool Last, int Opcode,
          typename SubTile, typename Parent0, typename SubTile0,
          typename Parent1, typename SubTile1>
PTO_REGION_ALWAYS_INLINE void pto_region_binary_assemble(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent0, SubTile0> &src0,
    region::SubTileView<Parent1, SubTile1> &src1) {
  static_assert(SubTile::BFractal == BLayout::RowMajor &&
                    SubTile0::IsCubeLayout && SubTile1::IsCubeLayout,
                "B.ASSEMBLE destination must be RowMajor and B.SUBVIEW "
                "sources must use CUBE layouts");
  static_assert(SubTile::SFractal == SLayout::NoneBox &&
                    SubTile0::SFractal == SLayout::NoneBox &&
                    SubTile1::SFractal == SLayout::NoneBox,
                "inline Tile region path requires unboxed fragments");
  static_assert(SubTile::Rows == SubTile0::Rows &&
                    SubTile::Cols == SubTile0::Cols &&
                    SubTile0::Rows == SubTile1::Rows &&
                    SubTile0::Cols == SubTile1::Cols,
                "binary TileArray slot requires matching physical shapes");
  static_assert(SubTile::ValidRow == SubTile0::ValidRow &&
                    SubTile::ValidCol == SubTile0::ValidCol &&
                    SubTile0::ValidRow == SubTile1::ValidRow &&
                    SubTile0::ValidCol == SubTile1::ValidCol,
                "binary TileArray slot requires matching valid shapes");
  static_assert(std::is_same_v<typename SubTile::DType,
                               typename SubTile0::DType> &&
                    std::is_same_v<typename SubTile0::DType,
                                   typename SubTile1::DType>,
                "binary TileArray slot requires matching element types");
  // PTO-ISA #265 (issue #702): field 5 is the writer extent in every phase;
  // the INIT destination B.IOT allocates and carries the parent capacity.
  constexpr int writer_size =
      tile_type_traits<typename SubTile::TileDType>::TilesizeCode;
  static_assert(writer_size != 0,
                "TileArray slot writer extent must be a nonzero size code");
  // The writer-fits-parent bound is guaranteed by TileArray construction:
  // ParentBytes = slot_count * SubTile::LogicalTileBytes. ParentSize here is
  // only the dispatch case label, and the switch instantiates every case,
  // so a compile-time comparison against it would spuriously fail.
  const uintptr_t source0_base_units = src0.GetRangeBase();
  const uintptr_t source1_base_units = src1.GetRangeBase();
  const uintptr_t destination_base_units = dst.range_base_units();
#define PTO_REGION_BINARY_ASSEMBLY_BODY                                     \
  "BSTART.TEPL %c11, %D1\n"                                               \
  "B.DIM zero, %c4, ->lb0\n"                                                   \
  "B.DIM zero, %c5, ->lb1\n"                                                   \
  "B.DIM zero, %c6, ->lb2\n"                                               \
  "B.IOT %2, %3, mask=1111, last, ->%0<%Z7>\n"                             \
  "B.SUBVIEW 0, %8, 0, %c12\n"                                             \
  "B.SUBVIEW 1, %9, 0, %c12\n"                                             \
  "B.ASSEMBLE %c14, %c15, %10, 0, %c13\n"
#define PTO_REGION_BINARY_CONT_BODY                                         \
  "BSTART.TEPL %c11, %D1\n"                                               \
  "B.DIM zero, %c4, ->lb0\n"                                                   \
  "B.DIM zero, %c5, ->lb1\n"                                                   \
  "B.DIM zero, %c6, ->lb2\n"                                               \
  "B.IOT %2, %3, mask=1111\n"                                             \
  "B.SUBVIEW 0, %8, 0, %c12\n"                                             \
  "B.SUBVIEW 1, %9, 0, %c12\n"                                             \
  "B.IOT %0, mask=1111, last\n"                                           \
  "B.ASSEMBLE %c14, %c15, %10, 0, %c13\n"
#define PTO_REGION_BINARY_ASSEMBLY_INPUTS                                  \
  "i"(type_traits<typename SubTile0::DType>::TypeCode),                    \
  "Tr"(src0.data()), "Tr"(src1.data()),                                   \
  "i"(std::remove_reference_t<decltype(src0)>::ValidCol),                  \
  "i"(std::remove_reference_t<decltype(src0)>::ValidRow),                  \
  "i"(SubTile0::Cols),                                                      \
  "i"(ParentSize),                                                           \
  "r"(source0_base_units), "r"(source1_base_units),                       \
  "r"(destination_base_units), "i"(Opcode),                              \
  "i"(tile_type_traits<typename SubTile0::TileDType>::TilesizeCode),       \
  "i"(writer_size), "i"(Init), "i"(Last)
  if constexpr (Init) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_BINARY_ASSEMBLY_BODY
                 : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                 : PTO_REGION_BINARY_ASSEMBLY_INPUTS
                 : "memory");
  } else {
    // PTO-ISA #265 Local continuation: the final source-form binder slot
    // carries the ParentRef (PTO-B-IOT-STREAM-001: the final effective
    // record has L=1). Three operands cannot share one B.IOT record, so
    // the math pair stays on an earlier L=0 record and the parent register
    // binds the final record with L=1.
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_BINARY_CONT_BODY
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_BINARY_ASSEMBLY_INPUTS
                 : "memory");
  }
#undef PTO_REGION_BINARY_ASSEMBLY_INPUTS
#undef PTO_REGION_BINARY_ASSEMBLY_BODY
}

template <int ParentSize, int Opcode, typename SubTile, typename Parent0,
          typename SubTile0, typename Parent1, typename SubTile1>
PTO_REGION_ALWAYS_INLINE void pto_region_binary_assemble_phase(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent0, SubTile0> &src0,
    region::SubTileView<Parent1, SubTile1> &src1) {
  if (dst.slot_count() == 1)
    pto_region_binary_assemble<ParentSize, true, true, Opcode>(dst, src0,
                                                               src1);
  else if (dst.ordinal() == 0)
    pto_region_binary_assemble<ParentSize, true, false, Opcode>(dst, src0,
                                                                src1);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_binary_assemble<ParentSize, false, true, Opcode>(dst, src0,
                                                                src1);
  else
    pto_region_binary_assemble<ParentSize, false, false, Opcode>(dst, src0,
                                                                 src1);
}

template <int Opcode, typename SubTile, typename Parent0, typename SubTile0,
          typename Parent1, typename SubTile1>
PTO_REGION_ALWAYS_INLINE void pto_region_binary_assemble_dispatch(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<Parent0, SubTile0> &src0,
    region::SubTileView<Parent1, SubTile1> &src1) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_BINARY_PARENT_CASE(N)                                    \
  case N:                                                                   \
    pto_region_binary_assemble_phase<N, Opcode>(dst, src0, src1);           \
    break
    PTO_REGION_BINARY_PARENT_CASE(1);
    PTO_REGION_BINARY_PARENT_CASE(2);
    PTO_REGION_BINARY_PARENT_CASE(3);
    PTO_REGION_BINARY_PARENT_CASE(4);
    PTO_REGION_BINARY_PARENT_CASE(5);
    PTO_REGION_BINARY_PARENT_CASE(6);
    PTO_REGION_BINARY_PARENT_CASE(7);
    PTO_REGION_BINARY_PARENT_CASE(8);
    PTO_REGION_BINARY_PARENT_CASE(9);
    PTO_REGION_BINARY_PARENT_CASE(10);
    PTO_REGION_BINARY_PARENT_CASE(11);
    PTO_REGION_BINARY_PARENT_CASE(12);
#undef PTO_REGION_BINARY_PARENT_CASE
  default:
    __builtin_trap();
  }
}

// Ordinary computed CUBE_M32 tiles can produce one fragment of a larger
// CUBE_M32 TileArray parent directly.  Unlike the overload above, these
// sources are complete tiles rather than B.SUBVIEW carriers, so the producer
// binds them directly and attaches only the destination B.ASSEMBLE modifier.
//
// INIT allocates the full parent capacity.  MIDDLE/LAST bind the existing
// ParentRef on the final L=1 record, while B.ASSEMBLE field 5 continues to
// describe the individual fragment writer extent in every phase.
template <int ParentSize, bool Init, bool Last, int Opcode, typename SubTile,
          typename Src0, typename Src1>
  requires(is_tile_data_v<Src0> && is_tile_data_v<Src1> &&
           !is_subtile_view_v<Src0> && !is_subtile_view_v<Src1> &&
           !is_range_v<Src0> && !is_range_v<Src1>)
PTO_REGION_ALWAYS_INLINE void pto_region_binary_tile_assemble(
    region::TileArrayOutputRef<SubTile> &dst, Src0 &src0, Src1 &src1) {
  static_assert(SubTile::BFractal == BLayout::CubeM32 &&
                    Src0::BFractal == BLayout::CubeM32 &&
                    Src1::BFractal == BLayout::CubeM32,
                "binary TileArray ordinary-tile assembly requires CUBE_M32 "
                "destination and sources");
  static_assert(SubTile::SFractal == SLayout::NoneBox &&
                    Src0::SFractal == SLayout::NoneBox &&
                    Src1::SFractal == SLayout::NoneBox,
                "binary TileArray ordinary-tile assembly requires unboxed "
                "fragments");
  static_assert(SubTile::Loc == Src0::Loc && Src0::Loc == Src1::Loc,
                "binary TileArray slot and sources must have matching "
                "locations");
  static_assert(SubTile::Rows == Src0::Rows && SubTile::Cols == Src0::Cols &&
                    Src0::Rows == Src1::Rows && Src0::Cols == Src1::Cols,
                "binary TileArray slot requires matching physical shapes");
  static_assert(SubTile::ValidRow == Src0::ValidRow &&
                    SubTile::ValidCol == Src0::ValidCol &&
                    Src0::ValidRow == Src1::ValidRow &&
                    Src0::ValidCol == Src1::ValidCol,
                "binary TileArray slot requires matching valid shapes");
  static_assert(std::is_same_v<typename SubTile::DType, typename Src0::DType> &&
                    std::is_same_v<typename Src0::DType, typename Src1::DType>,
                "binary TileArray slot requires matching element types");
  static_assert(SubTile::LogicalTileBytes == Src0::LogicalTileBytes &&
                    Src0::LogicalTileBytes == Src1::LogicalTileBytes,
                "binary TileArray slot and sources must have matching "
                "capacities");

  constexpr int writer_size =
      tile_type_traits<typename SubTile::TileDType>::TilesizeCode;
  static_assert(writer_size != 0,
                "TileArray slot writer extent must be a nonzero size code");
  const uintptr_t destination_base_units = dst.range_base_units();

#define PTO_REGION_BINARY_TILE_ASSEMBLY_PREFIX                                 \
  "BSTART.TEPL %c[Opcode], %D[Type]\n" PTO_REGION_ELEMENTWISE_LAYOUT_ASM       \
  "B.DIM zero, %c[ValidCol], ->lb0\n"                                          \
  "B.DIM zero, %c[ValidRow], ->lb1\n"                                          \
  "B.DIM zero, %c[Cols], ->lb2\n"
#define PTO_REGION_BINARY_TILE_ASSEMBLY_INPUTS                                 \
  [Type] "i"(type_traits<typename Src0::DType>::TypeCode),                     \
      [Src0] "Tr"(src0.data()), [Src1] "Tr"(src1.data()),                      \
      [ValidCol] "i"(Src0::ValidCol), [ValidRow] "i"(Src0::ValidRow),          \
      [Cols] "i"(Src0::Cols), [ParentSize] "i"(ParentSize),                    \
      [Offset] "r"(destination_base_units), [Opcode] "i"(Opcode),              \
      [WriterSize] "i"(writer_size), [Init] "i"(Init), [Last] "i"(Last),       \
      [ElemLayout] "i"(local_layout_code_v<SubTile>)
  if constexpr (Init) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_BINARY_TILE_ASSEMBLY_PREFIX
                 "B.IOT %[Src0], %[Src1], mask=1111, last, "
                 "->%[Dst]<%Z[ParentSize]>\n"
                 "B.ASSEMBLE %c[Init], %c[Last], %[Offset], 0, %c[WriterSize]\n"
                 : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
                 : PTO_REGION_BINARY_TILE_ASSEMBLY_INPUTS
                 : "memory");
  } else {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_BINARY_TILE_ASSEMBLY_PREFIX
                 "B.IOT %[Src0], %[Src1], mask=1111\n"
                 "B.IOT %[Dst], mask=1111, last\n"
                 "B.ASSEMBLE %c[Init], %c[Last], %[Offset], 0, %c[WriterSize]\n"
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_BINARY_TILE_ASSEMBLY_INPUTS
                 : "memory");
  }
#undef PTO_REGION_BINARY_TILE_ASSEMBLY_INPUTS
#undef PTO_REGION_BINARY_TILE_ASSEMBLY_PREFIX
}

template <int ParentSize, int Opcode, typename SubTile, typename Src0,
          typename Src1>
  requires(is_tile_data_v<Src0> && is_tile_data_v<Src1> &&
           !is_subtile_view_v<Src0> && !is_subtile_view_v<Src1> &&
           !is_range_v<Src0> && !is_range_v<Src1>)
PTO_REGION_ALWAYS_INLINE void pto_region_binary_tile_assemble_phase(
    region::TileArrayOutputRef<SubTile> &dst, Src0 &src0, Src1 &src1) {
  if (dst.slot_count() == 1)
    pto_region_binary_tile_assemble<ParentSize, true, true, Opcode>(dst, src0,
                                                                    src1);
  else if (dst.ordinal() == 0)
    pto_region_binary_tile_assemble<ParentSize, true, false, Opcode>(dst, src0,
                                                                     src1);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_binary_tile_assemble<ParentSize, false, true, Opcode>(dst, src0,
                                                                     src1);
  else
    pto_region_binary_tile_assemble<ParentSize, false, false, Opcode>(dst, src0,
                                                                      src1);
}

template <int Opcode, typename SubTile, typename Src0, typename Src1>
  requires(is_tile_data_v<Src0> && is_tile_data_v<Src1> &&
           !is_subtile_view_v<Src0> && !is_subtile_view_v<Src1> &&
           !is_range_v<Src0> && !is_range_v<Src1>)
PTO_REGION_ALWAYS_INLINE void pto_region_binary_tile_assemble_dispatch(
    region::TileArrayOutputRef<SubTile> &dst, Src0 &src0, Src1 &src1) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_BINARY_TILE_PARENT_CASE(N)                                  \
  case N:                                                                      \
    pto_region_binary_tile_assemble_phase<N, Opcode>(dst, src0, src1);         \
    break
    PTO_REGION_BINARY_TILE_PARENT_CASE(1);
    PTO_REGION_BINARY_TILE_PARENT_CASE(2);
    PTO_REGION_BINARY_TILE_PARENT_CASE(3);
    PTO_REGION_BINARY_TILE_PARENT_CASE(4);
    PTO_REGION_BINARY_TILE_PARENT_CASE(5);
    PTO_REGION_BINARY_TILE_PARENT_CASE(6);
    PTO_REGION_BINARY_TILE_PARENT_CASE(7);
    PTO_REGION_BINARY_TILE_PARENT_CASE(8);
    PTO_REGION_BINARY_TILE_PARENT_CASE(9);
    PTO_REGION_BINARY_TILE_PARENT_CASE(10);
    PTO_REGION_BINARY_TILE_PARENT_CASE(11);
    PTO_REGION_BINARY_TILE_PARENT_CASE(12);
#undef PTO_REGION_BINARY_TILE_PARENT_CASE
  default:
    __builtin_trap();
  }
}

// A mixed binary assembly producer binds one parent-backed CUBE_M32 fragment
// through B.SUBVIEW and one ordinary CUBE_M32 Tile directly. The bool records
// the mathematical source position of the view so B.SUBVIEW selects the same
// source independently of the C++ parameter order used internally here.
template <int ParentSize, bool Init, bool Last, int Opcode, bool ViewIsSrc0,
          typename SubTile, typename SourceParent, typename SourceSubTile,
          typename Tile>
  requires(is_tile_data_v<Tile> && !is_subtile_view_v<Tile> &&
           !is_range_v<Tile>)
PTO_REGION_ALWAYS_INLINE void pto_region_binary_mixed_assemble(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<SourceParent, SourceSubTile> &view, Tile &tile) {
  static_assert(SubTile::BFractal == BLayout::CubeM32 &&
                    SourceSubTile::BFractal == BLayout::CubeM32 &&
                    Tile::BFractal == BLayout::CubeM32,
                "binary TileArray mixed-source assembly requires CUBE_M32 "
                "destination and sources");
  static_assert(SubTile::SFractal == SLayout::NoneBox &&
                    SourceSubTile::SFractal == SLayout::NoneBox &&
                    Tile::SFractal == SLayout::NoneBox,
                "binary TileArray mixed-source assembly requires unboxed "
                "fragments");
  static_assert(SubTile::Loc == SourceSubTile::Loc &&
                    SourceSubTile::Loc == Tile::Loc,
                "binary TileArray slot and mixed sources must have matching "
                "locations");
  static_assert(SubTile::Rows == SourceSubTile::Rows &&
                    SubTile::Cols == SourceSubTile::Cols &&
                    SourceSubTile::Rows == Tile::Rows &&
                    SourceSubTile::Cols == Tile::Cols,
                "binary TileArray slot requires matching physical shapes");
  static_assert(SubTile::ValidRow == SourceSubTile::ValidRow &&
                    SubTile::ValidCol == SourceSubTile::ValidCol &&
                    SourceSubTile::ValidRow == Tile::ValidRow &&
                    SourceSubTile::ValidCol == Tile::ValidCol,
                "binary TileArray slot requires matching valid shapes");
  static_assert(
      std::is_same_v<typename SubTile::DType,
                     typename SourceSubTile::DType> &&
          std::is_same_v<typename SourceSubTile::DType, typename Tile::DType>,
      "binary TileArray slot requires matching element types");
  static_assert(SubTile::LogicalTileBytes ==
                        SourceSubTile::LogicalTileBytes &&
                    SourceSubTile::LogicalTileBytes == Tile::LogicalTileBytes,
                "binary TileArray slot and mixed sources must have matching "
                "capacities");

  constexpr int writer_size =
      tile_type_traits<typename SubTile::TileDType>::TilesizeCode;
  constexpr int subview_size =
      tile_type_traits<typename SourceSubTile::TileDType>::TilesizeCode;
  static_assert(writer_size != 0,
                "TileArray slot writer extent must be a nonzero size code");
  static_assert(subview_size != 0,
                "mixed-source B.SUBVIEW extent must be a nonzero size code");
  const uintptr_t source_base_units = view.GetRangeBase();
  const uintptr_t destination_base_units = dst.range_base_units();

#define PTO_REGION_BINARY_MIXED_ASSEMBLY_PREFIX                               \
  "BSTART.TEPL %c[Opcode], %D[Type]\n" PTO_REGION_ELEMENTWISE_LAYOUT_ASM     \
  "B.DIM zero, %c[ValidCol], ->lb0\n"                                        \
  "B.DIM zero, %c[ValidRow], ->lb1\n"                                        \
  "B.DIM zero, %c[Cols], ->lb2\n"
#define PTO_REGION_BINARY_MIXED_ASSEMBLY_INPUTS                               \
  [Type] "i"(type_traits<typename SourceSubTile::DType>::TypeCode),           \
      [View] "Tr"(view.data()), [Tile] "Tr"(tile.data()),                     \
      [ValidCol] "i"(SourceSubTile::ValidCol),                                \
      [ValidRow] "i"(SourceSubTile::ValidRow),                                \
      [Cols] "i"(SourceSubTile::Cols), [ParentSize] "i"(ParentSize),          \
      [SourceOffset] "r"(source_base_units),                                  \
      [DestinationOffset] "r"(destination_base_units), [Opcode] "i"(Opcode), \
      [SubviewSize] "i"(subview_size), [WriterSize] "i"(writer_size),        \
      [Init] "i"(Init), [Last] "i"(Last),                                    \
      [ElemLayout] "i"(local_layout_code_v<SubTile>)
  if constexpr (Init && ViewIsSrc0) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
        PTO_REGION_BINARY_MIXED_ASSEMBLY_PREFIX
        "B.IOT %[View], %[Tile], mask=1111, last, "
        "->%[Dst]<%Z[ParentSize]>\n"
        "B.SUBVIEW 0, %[SourceOffset], 0, %c[SubviewSize]\n"
        "B.ASSEMBLE %c[Init], %c[Last], %[DestinationOffset], 0, "
        "%c[WriterSize]\n"
        : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
        : PTO_REGION_BINARY_MIXED_ASSEMBLY_INPUTS
        : "memory");
  } else if constexpr (Init) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(
        PTO_REGION_BINARY_MIXED_ASSEMBLY_PREFIX
        "B.IOT %[Tile], %[View], mask=1111, last, "
        "->%[Dst]<%Z[ParentSize]>\n"
        "B.SUBVIEW 1, %[SourceOffset], 0, %c[SubviewSize]\n"
        "B.ASSEMBLE %c[Init], %c[Last], %[DestinationOffset], 0, "
        "%c[WriterSize]\n"
        : [Dst] "=Tr"(dst.template parent_data<ParentSize>())
        : PTO_REGION_BINARY_MIXED_ASSEMBLY_INPUTS
        : "memory");
  } else if constexpr (ViewIsSrc0) {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_BINARY_MIXED_ASSEMBLY_PREFIX
                 "B.IOT %[View], %[Tile], mask=1111\n"
                 "B.SUBVIEW 0, %[SourceOffset], 0, %c[SubviewSize]\n"
                 "B.IOT %[Dst], mask=1111, last\n"
                 "B.ASSEMBLE %c[Init], %c[Last], "
                 "%[DestinationOffset], 0, %c[WriterSize]\n"
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_BINARY_MIXED_ASSEMBLY_INPUTS
                 : "memory");
  } else {
    /* PTO_LINX_GUARD */ ::pto::linx_experimental::region<decltype(dst)>(); /* PTO_LINX_GUARD_END */ asm volatile(PTO_REGION_BINARY_MIXED_ASSEMBLY_PREFIX
                 "B.IOT %[Tile], %[View], mask=1111\n"
                 "B.SUBVIEW 1, %[SourceOffset], 0, %c[SubviewSize]\n"
                 "B.IOT %[Dst], mask=1111, last\n"
                 "B.ASSEMBLE %c[Init], %c[Last], "
                 "%[DestinationOffset], 0, %c[WriterSize]\n"
                 :
                 : [Dst] "Tr"(dst.template parent_data<ParentSize>()),
                   PTO_REGION_BINARY_MIXED_ASSEMBLY_INPUTS
                 : "memory");
  }
#undef PTO_REGION_BINARY_MIXED_ASSEMBLY_INPUTS
#undef PTO_REGION_BINARY_MIXED_ASSEMBLY_PREFIX
}

template <int ParentSize, int Opcode, bool ViewIsSrc0, typename SubTile,
          typename SourceParent, typename SourceSubTile, typename Tile>
  requires(is_tile_data_v<Tile> && !is_subtile_view_v<Tile> &&
           !is_range_v<Tile>)
PTO_REGION_ALWAYS_INLINE void pto_region_binary_mixed_assemble_phase(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<SourceParent, SourceSubTile> &view, Tile &tile) {
  if (dst.slot_count() == 1)
    pto_region_binary_mixed_assemble<ParentSize, true, true, Opcode,
                                     ViewIsSrc0>(dst, view, tile);
  else if (dst.ordinal() == 0)
    pto_region_binary_mixed_assemble<ParentSize, true, false, Opcode,
                                     ViewIsSrc0>(dst, view, tile);
  else if (dst.ordinal() == dst.slot_count() - 1)
    pto_region_binary_mixed_assemble<ParentSize, false, true, Opcode,
                                     ViewIsSrc0>(dst, view, tile);
  else
    pto_region_binary_mixed_assemble<ParentSize, false, false, Opcode,
                                     ViewIsSrc0>(dst, view, tile);
}

template <int Opcode, bool ViewIsSrc0, typename SubTile, typename SourceParent,
          typename SourceSubTile, typename Tile>
  requires(is_tile_data_v<Tile> && !is_subtile_view_v<Tile> &&
           !is_range_v<Tile>)
PTO_REGION_ALWAYS_INLINE void pto_region_binary_mixed_assemble_dispatch(
    region::TileArrayOutputRef<SubTile> &dst,
    region::SubTileView<SourceParent, SourceSubTile> &view, Tile &tile) {
  switch (dst.parent_size_code()) {
#define PTO_REGION_BINARY_MIXED_PARENT_CASE(N)                                \
  case N:                                                                     \
    pto_region_binary_mixed_assemble_phase<N, Opcode, ViewIsSrc0>(dst, view,  \
                                                                   tile);     \
    break
    PTO_REGION_BINARY_MIXED_PARENT_CASE(1);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(2);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(3);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(4);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(5);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(6);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(7);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(8);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(9);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(10);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(11);
    PTO_REGION_BINARY_MIXED_PARENT_CASE(12);
#undef PTO_REGION_BINARY_MIXED_PARENT_CASE
  default:
    __builtin_trap();
  }
}

#define PTO_REGION_BINARY_DEST_WRAPPER(Name, Opcode)                           \
  template <typename SubTile, typename Parent0, typename SubTile0,             \
            typename Parent1, typename SubTile1>                               \
  PTO_REGION_ALWAYS_INLINE void Name(                                          \
      region::TileArrayOutputRef<SubTile> dst,                                 \
      region::SubTileView<Parent0, SubTile0> &src0,                            \
      region::SubTileView<Parent1, SubTile1> &src1) {                          \
    pto_region_binary_assemble_dispatch<Opcode>(dst, src0, src1);              \
  }                                                                            \
  template <typename SubTile, typename Src0, typename Src1>                    \
    requires(is_tile_data_v<Src0> && is_tile_data_v<Src1> &&                   \
             !is_subtile_view_v<Src0> && !is_subtile_view_v<Src1> &&           \
             !is_range_v<Src0> && !is_range_v<Src1>)                           \
  PTO_REGION_ALWAYS_INLINE void Name(region::TileArrayOutputRef<SubTile> dst,  \
                                     Src0 &src0, Src1 &src1) {                 \
    pto_region_binary_tile_assemble_dispatch<Opcode>(dst, src0, src1);         \
  }                                                                            \
  template <typename SubTile, typename SourceParent, typename SourceSubTile,   \
            typename Tile>                                                     \
    requires(is_tile_data_v<Tile> && !is_subtile_view_v<Tile> &&               \
             !is_range_v<Tile>)                                                \
  PTO_REGION_ALWAYS_INLINE void Name(                                          \
      region::TileArrayOutputRef<SubTile> dst,                                 \
      region::SubTileView<SourceParent, SourceSubTile> &src0, Tile &src1) {    \
    pto_region_binary_mixed_assemble_dispatch<Opcode, true>(dst, src0, src1);  \
  }                                                                            \
  template <typename SubTile, typename Tile, typename SourceParent,            \
            typename SourceSubTile>                                            \
    requires(is_tile_data_v<Tile> && !is_subtile_view_v<Tile> &&               \
             !is_range_v<Tile>)                                                \
  PTO_REGION_ALWAYS_INLINE void Name(                                          \
      region::TileArrayOutputRef<SubTile> dst, Tile &src0,                     \
      region::SubTileView<SourceParent, SourceSubTile> &src1) {                \
    pto_region_binary_mixed_assemble_dispatch<Opcode, false>(dst, src1, src0); \
  }

PTO_REGION_BINARY_DEST_WRAPPER(TADD, 0)
PTO_REGION_BINARY_DEST_WRAPPER(TSUB, 1)
PTO_REGION_BINARY_DEST_WRAPPER(TMUL, 2)
PTO_REGION_BINARY_DEST_WRAPPER(TDIV, 3)
PTO_REGION_BINARY_DEST_WRAPPER(TREM, 4)
PTO_REGION_BINARY_DEST_WRAPPER(TAND, 6)
PTO_REGION_BINARY_DEST_WRAPPER(TOR, 7)
PTO_REGION_BINARY_DEST_WRAPPER(TXOR, 8)
PTO_REGION_BINARY_DEST_WRAPPER(TSHL, 9)
PTO_REGION_BINARY_DEST_WRAPPER(TSHR, 10)
PTO_REGION_BINARY_DEST_WRAPPER(TMAX, 11)
PTO_REGION_BINARY_DEST_WRAPPER(TMIN, 12)

#undef PTO_REGION_BINARY_DEST_WRAPPER

#undef PTO_REGION_ALWAYS_INLINE

} // namespace pto

#endif
