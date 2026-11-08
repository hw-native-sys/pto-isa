#ifndef PTO_TILEOP_HPP
#define PTO_TILEOP_HPP

#include "pto/npu/linx/impl/common/pto_tileop_api_revision.hpp"
#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include "pto/npu/linx/impl/common/pto_shared_nd.hpp"
#include "pto/npu/linx/impl/common/pto_indexed_range.hpp"
#include "pto/npu/linx/impl/common/pto_tile_region.hpp"
#include "pto/npu/linx/impl/common/tileop_api.hpp"
#include "pto/npu/linx/impl/common/global_iterator.hpp"
#include "pto/npu/linx/impl/common/tile_tensor_impl.hpp"
#include "pto/npu/linx/impl/common/debug_utils.hpp"

#ifdef __linx
#include "pto/npu/linx/impl/common/pto_tile_region_inline_asm.hpp"
#include "pto/npu/linx/impl/common/pto_predicate.hpp"
#endif

// Returns the current PE ID (0..3).
// Aligned with website manual get_thread_idx(). Lowered to SSR_GET reading
// the read-only PEID SSR (0x0802).
#ifdef __cpu_sim__
inline uint32_t get_thread_idx() {
  return 0;
}
#else
inline uint32_t get_thread_idx() {
  return __builtin_linx_get_thread_idx();
}
#endif

// API convenience alias; this is not a separate architectural operation.
inline uint32_t get_thread_id() {
  return get_thread_idx();
}

#endif
