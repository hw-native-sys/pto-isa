#ifndef _INCLUDE_TYPE_H_
#define _INCLUDE_TYPE_H_

template <typename E> struct type_traits {
  static constexpr int bits = sizeof(E) * 8;
};

#ifdef __linx
#include <pto/npu/linx/impl/jcore/type.hpp>
#elif defined(__cpu_sim__)
// The Linx CPU simulator uses the same architectural type codes as the Linx
// compiler. Pull in the host storage fallback together with those codes so
// common Tile traits stay identical between NPU and CPU-sim builds.
#include <pto/npu/linx/impl/jcore/type.hpp>
#elif defined(__ARM_FEATURE_SME)
#include <pto/npu/linx/impl/aarch64/type.hpp>
#elif defined(__accelerator)
#include <accelerator/v220/type.hpp>
#else
#error "__linx, __cpu_sim__, __ARM_FEATURE_SME, or __accelerator must be defined"
#endif

#endif
