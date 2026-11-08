#ifndef PTO_JCORE_LINX_COMPAT_TYPES_HPP
#define PTO_JCORE_LINX_COMPAT_TYPES_HPP

#include <stdint.h>

// On the LLVM Linx target (__linx), the scalar storage types (__tf32 etc.)
// are provided by the compiler's linx_blkc.h builtin header, so this compat
// fallback must not redefine them.
#if defined(__linx)
#define PTO_LINX_COMPAT_TYPES_PROVIDED 1

// RCPE6M2 (pto-spec #322, TileOP #165) is a source-only derived type with no
// scalar FCVT encoding, so blkc.h historically did not spell it. It now does
// (LinxISA/llvm-project "[clang] Declare __fp8_rcpe6m2 scalar type", PR #97):
// -mlxbc force-includes linx_blkc.h before every TileOP header.  Older target
// toolchains define __LINX_BLKC but do not yet declare this type, so the
// header-presence macro is not sufficient to decide whether a fallback is
// needed.  The scalar type's builtin declaration also publishes its usual
// `*_TYPE` marker; use that marker to keep the fallback out of newer headers.
#if !defined(__fp8_rcpe6m2_TYPE) && !defined(PTO_LINX_HOST_CXX)
struct __fp8_rcpe6m2 { uint8_t data; };
#endif
#elif !defined(PTO_LINX_COMPAT_TYPES_PROVIDED)

// The public TileOP surface names scalar storage formats that are not C++
// fundamental types.  Keep those names available to the Linx target while the
// architectural operation encoding continues to be selected by type_traits.
using __fp32 = float;
using __half = _Float16;

#define PTO_LINX_STORAGE_TYPE(NAME, STORAGE)                                  \
  struct NAME { STORAGE data; }

PTO_LINX_STORAGE_TYPE(__tf32, uint32_t);
PTO_LINX_STORAGE_TYPE(__hf32, uint32_t);
PTO_LINX_STORAGE_TYPE(__blkc_bf16, uint16_t);
PTO_LINX_STORAGE_TYPE(__hif8, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp8_e4m3, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp8_e5m2, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp8_e6m2, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp8_rcpe6m2, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp6_e3m2, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp6_e2m3, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp4_e2m1x2, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp4_e1m2x2, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp8_e8m0, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp4_hif4x2, uint8_t);
PTO_LINX_STORAGE_TYPE(__int4x2, int8_t);
PTO_LINX_STORAGE_TYPE(__uint4x2, uint8_t);
PTO_LINX_STORAGE_TYPE(__fp16x2, uint32_t);
PTO_LINX_STORAGE_TYPE(__bf16x2, uint32_t);
PTO_LINX_STORAGE_TYPE(__uint16x2, uint32_t);
PTO_LINX_STORAGE_TYPE(__int16x2, uint32_t);
PTO_LINX_STORAGE_TYPE(__fp8_e4m3x4, uint32_t);
PTO_LINX_STORAGE_TYPE(__fp8_e5m2x4, uint32_t);
PTO_LINX_STORAGE_TYPE(__uint8x4, uint32_t);
PTO_LINX_STORAGE_TYPE(__int8x4, uint32_t);
PTO_LINX_STORAGE_TYPE(__fp8_e6m2x2, uint16_t);
PTO_LINX_STORAGE_TYPE(__fp8_e4m3x2, uint16_t);
PTO_LINX_STORAGE_TYPE(__fp8_e5m2x2, uint16_t);

#undef PTO_LINX_STORAGE_TYPE

#define __tf32_STORAGE(value) ((value).data)
#define __hf32_STORAGE(value) ((value).data)
#define __blkc_bf16_STORAGE(value) ((value).data)
#define __hif8_STORAGE(value) ((value).data)
#define __fp8_e4m3_STORAGE(value) ((value).data)
#define __fp8_e5m2_STORAGE(value) ((value).data)
#define __fp8_e6m2_STORAGE(value) ((value).data)
#define __fp8_rcpe6m2_STORAGE(value) ((value).data)
#define __fp6_e3m2_STORAGE(value) ((value).data)
#define __fp6_e2m3_STORAGE(value) ((value).data)
#define __fp4_e2m1x2_STORAGE(value) ((value).data)
#define __fp4_e1m2x2_STORAGE(value) ((value).data)
#define __fp8_e8m0_STORAGE(value) ((value).data)
#define __fp4_hif4x2_STORAGE(value) ((value).data)
#define __int4x2_STORAGE(value) ((value).data)
#define __uint4x2_STORAGE(value) ((value).data)
#define __fp16x2_STORAGE(value) ((value).data)
#define __bf16x2_STORAGE(value) ((value).data)
#define __uint16x2_STORAGE(value) ((value).data)
#define __int16x2_STORAGE(value) ((value).data)
#define __fp8_e4m3x4_STORAGE(value) ((value).data)
#define __fp8_e5m2x4_STORAGE(value) ((value).data)
#define __uint8x4_STORAGE(value) ((value).data)
#define __int8x4_STORAGE(value) ((value).data)
#define __fp8_e6m2x2_STORAGE(value) ((value).data)
#define __fp8_e4m3x2_STORAGE(value) ((value).data)
#define __fp8_e5m2x2_STORAGE(value) ((value).data)

#endif

#endif
