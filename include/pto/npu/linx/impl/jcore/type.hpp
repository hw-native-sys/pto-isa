/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_LINX_NATIVE__INCLUDE_JCORE_TYPE_H_
#define PTO_LINX_NATIVE__INCLUDE_JCORE_TYPE_H_

#include <type_traits>
#include <cstddef>

#include "pto/npu/linx/impl/jcore/linx_compat_types.hpp"

enum __type_code {
  __type_fp64 = 0,
  __type_fp32 = 1,
  __type_tf32 = 2,
  __type_hf32 = 3,

  __type_fp16 = 4,
  __type_bf16 = 5,
  __type_hif8 = 6,

  __type_fp8_e4m3 = 7,
  __type_fp8_e5m2 = 8,
  __type_fp6_e3m2 = 9,
  __type_fp5_e2m3 = 10,

  __type_fp4_e2m1x2 = 11,
  __type_fp4_e1m2x2 = 12,
  __type_fp8_e8m0 = 13,

  __type_fp4_hif4x2 = 14,
  __type_fp8_e6m2 = 15,

  __type_int64 = 16,
  __type_int32 = 17,
  __type_int16 = 18,
  __type_int8 = 19,
  __type_int4x2 = 20,
  __type_rcpe6m2 = 21,

  __type_uint64 = 24,
  __type_uint32 = 25,
  __type_uint16 = 26,
  __type_uint8 = 27,
  __type_uint4x2 = 28,
};

template <int C, int b> struct type_traits_base {
  static constexpr int TypeCode = C;
  static constexpr int bits = b;
};

// clang-format off
template<> struct type_traits<double>         : public type_traits_base<__type_fp64, 64> {};
template<> struct type_traits<__fp32>         : public type_traits_base<__type_fp32, 32> {};
template<> struct type_traits<__tf32>         : public type_traits_base<__type_tf32, 32> {};
template<> struct type_traits<__hf32>         : public type_traits_base<__type_hf32, 32> {};

template<> struct type_traits<__half>         : public type_traits_base<__type_fp16, 16> {};
template<> struct type_traits<__bf16>         : public type_traits_base<__type_bf16, 16> {};
template<> struct type_traits<__hif8>         : public type_traits_base<__type_hif8, 8> {};

template<> struct type_traits<__fp8_e4m3>     : public type_traits_base<__type_fp8_e4m3, 8> {};
template<> struct type_traits<__fp8_e5m2>     : public type_traits_base<__type_fp8_e5m2, 8> {};
template<> struct type_traits<__fp6_e3m2>     : public type_traits_base<__type_fp6_e3m2, 6> {};
template<> struct type_traits<__fp6_e2m3>     : public type_traits_base<__type_fp5_e2m3, 6> {};
template<> struct type_traits<__fp4_e2m1x2>   : public type_traits_base<__type_fp4_e2m1x2, 8> {};
template<> struct type_traits<__fp4_e1m2x2>   : public type_traits_base<__type_fp4_e1m2x2, 8> {};
template<> struct type_traits<__fp8_e8m0>     : public type_traits_base<__type_fp8_e8m0, 8> {};
template<> struct type_traits<__fp4_hif4x2>   : public type_traits_base<__type_fp4_hif4x2, 8> {};
template<> struct type_traits<__fp8_e6m2>     : public type_traits_base<__type_fp8_e6m2, 8> {};
template<> struct type_traits<__fp8_rcpe6m2>  : public type_traits_base<__type_rcpe6m2, 8> {};

template<> struct type_traits<int64_t>        : public type_traits_base<__type_int64, 64> {};
template<> struct type_traits<int32_t>        : public type_traits_base<__type_int32, 32> {};
template<> struct type_traits<int16_t>        : public type_traits_base<__type_int16, 16> {};
template<> struct type_traits<int8_t>         : public type_traits_base<__type_int8, 8> {};
template<> struct type_traits<__int4x2>       : public type_traits_base<__type_int4x2, 8> {};

template<> struct type_traits<unsigned long>  : public type_traits_base<__type_uint64, 64> {};
template<> struct type_traits<unsigned long long> : public type_traits_base<__type_uint64, 64> {};
template<> struct type_traits<unsigned int>   : public type_traits_base<__type_uint32, 32> {};
template<> struct type_traits<unsigned short> : public type_traits_base<__type_uint16, 16> {};
template<> struct type_traits<unsigned char>  : public type_traits_base<__type_uint8, 8> {};
template<> struct type_traits<__uint4x2>      : public type_traits_base<__type_uint4x2, 8> {};
// clang-format on


// SizeCode is a 4-bit code encoding the per-PE tile size (PTO-ISA ADR 0069,
// commit 1e91bf9). There is no 4-PE (whole-core) multiplier: the developer's
// byte count is the size that flows directly into the encoded SizeCode.
//   B.IOT (Local) destination: 1..12 = 128 B..256 KB per PE
//   B.IOS (Shared) destination: 1..12 = 128 B..256 KB per PE
//   0 is the source-only encoding; 13..15 reserved.
enum __tilesize_code {
  __tilesize_implicit = 0,
  __tilesize_128B = 1,
  __tilesize_256B = 2,
  __tilesize_512B = 3,
  __tilesize_1KB = 4,
  __tilesize_2KB = 5,
  __tilesize_4KB = 6,
  __tilesize_8KB = 7,
  __tilesize_16KB = 8,
  __tilesize_32KB = 9,
  __tilesize_64KB = 10,
  __tilesize_128KB = 11,
  __tilesize_256KB = 12,
  __tilesize_unknown = -1
};

// PTO ISA 0.58.3 encodes a three-bit PEMode, not an arbitrary four-bit
// participation mask.  Keep masks at the C++ API boundary because they are
// easier to read in call sites, but reject every mask which the fixed decoder
// cannot produce.
constexpr bool is_valid_pe_mask(unsigned mask) {
  return mask == 0 || mask == 8 || mask == 4 || mask == 2 || mask == 1 ||
         mask == 12 || mask == 14 || mask == 15;
}

// GMOV consumes the raw four-bit PE mask.  Unlike the legacy PE-mode users,
// its 0.58.6 contract accepts every nonzero mask; zero is handled by GMOV as
// its architectural strict no-op.  Keep this predicate separate so widening
// GMOV does not accidentally change older APIs that still encode PEMode.
constexpr bool is_valid_gmov_pe_mask(unsigned mask) {
  return mask >= 1 && mask <= 15;
}

constexpr unsigned pe_mode_from_mask(unsigned mask) {
  return mask == 0  ? 0 :
         mask == 8  ? 1 :
         mask == 4  ? 2 :
         mask == 2  ? 3 :
         mask == 1  ? 4 :
         mask == 12 ? 5 :
         mask == 14 ? 6 :
         mask == 15 ? 7 : 8; // 8 is an invalid sentinel, never encodable.
}

template <typename T>
struct tile_type_traits {
private:
  using PlainT = std::remove_cv_t<std::remove_reference_t<T>>;
  // sizeof(PlainT) is the per-PE tile size in bytes; there is no 4-PE
  // multiplier, so the developer's byte count encodes directly.
  static constexpr std::size_t PETileBytes = sizeof(PlainT);

  static constexpr int mapBytesToEnum(std::size_t b) {
    return
      b == 128    ? __tilesize_128B :
      b == 256    ? __tilesize_256B :
      b == 512    ? __tilesize_512B :
      b == 1024   ? __tilesize_1KB  :
      b == 2048   ? __tilesize_2KB  :
      b == 4096   ? __tilesize_4KB  :
      b == 8192   ? __tilesize_8KB  :
      b == 16384  ? __tilesize_16KB :
      b == 32768  ? __tilesize_32KB :
      b == 65536  ? __tilesize_64KB :
      b == 131072 ? __tilesize_128KB:
      b == 262144 ? __tilesize_256KB:
      __tilesize_unknown;
  }

public:
  static constexpr int TilesizeCode = mapBytesToEnum(PETileBytes);
  static constexpr int Regsize = PETileBytes;
  // B.IOT (Local) destination capacity: 1..12 (128 B..256 KB per PE).
  static constexpr bool IsValidActiveSize =
      TilesizeCode >= __tilesize_128B && TilesizeCode <= __tilesize_256KB;
  // B.IOS (Shared) destination capacity: 1..12 (128 B..256 KB).
  static constexpr bool IsValidSharedActiveSize =
      TilesizeCode >= __tilesize_128B && TilesizeCode <= __tilesize_256KB;
};

#ifdef __linx
// Linx inline-asm Tile operands use one whole-register carrier regardless of
// their encoded SizeCode. Keep LogicalBytes in the wrapper type so
// tile_type_traits continues to project the architectural SizeCode. Scalar is
// uint32_t by default to preserve the established Tile ABI; the formal typed
// ElementTile profiles select their own 32-bit scalar carrier.
template <int LogicalBytes, typename Scalar = uint32_t>
struct linx_tile_carrier {
  // The on-object placeholder must cover the full logical tile capacity so
  // that stores/loads of the Tile object (including stack spills) transfer
  // the whole tile. The size is the logical capacity in 32-bit elements
  // (128 B -> 32 .. 256 KB -> 65536); 1024 elements only covers 4 KB and
  // silently truncated larger tiles (toolchain-build #14).
  static_assert(LogicalBytes % sizeof(Scalar) == 0,
                "tile carrier capacity must contain a whole number of elements");
  static constexpr unsigned kElements = LogicalBytes / sizeof(Scalar);
  using ScalarType = Scalar;
  using RegisterType = Scalar tile_size(kElements);
  RegisterType Register;
};

template <int LogicalBytes, typename Scalar>
struct tile_type_traits<linx_tile_carrier<LogicalBytes, Scalar>> {
private:
  static constexpr int mapBytesToEnum(int Bytes) {
    return
      Bytes == 128    ? __tilesize_128B :
      Bytes == 256    ? __tilesize_256B :
      Bytes == 512    ? __tilesize_512B :
      Bytes == 1024   ? __tilesize_1KB :
      Bytes == 2048   ? __tilesize_2KB :
      Bytes == 4096   ? __tilesize_4KB :
      Bytes == 8192   ? __tilesize_8KB :
      Bytes == 16384  ? __tilesize_16KB :
      Bytes == 32768  ? __tilesize_32KB :
      Bytes == 65536  ? __tilesize_64KB :
      Bytes == 131072 ? __tilesize_128KB :
      Bytes == 262144 ? __tilesize_256KB :
      __tilesize_unknown;
  }

public:
  static constexpr int TilesizeCode = mapBytesToEnum(LogicalBytes);
  static constexpr int Regsize = 4096;
  static constexpr bool IsValidActiveSize =
      TilesizeCode >= __tilesize_128B && TilesizeCode <= __tilesize_256KB;
  static constexpr bool IsValidSharedActiveSize =
      TilesizeCode >= __tilesize_128B && TilesizeCode <= __tilesize_256KB;
};
#endif

#endif
