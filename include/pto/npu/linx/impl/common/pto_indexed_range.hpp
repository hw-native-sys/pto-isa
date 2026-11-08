#ifndef PTO_INDEXED_RANGE_HPP
#define PTO_INDEXED_RANGE_HPP

#include <cstdint>

namespace pto {
// PTO-ISA PR #424 at 1981bceffa82. Bounds are inclusive signed byte
// displacements of access starts, relative to the operation's GM base.
// Software certifies coverage and a nonwrapping address envelope. This is
// not an element count, row stride, or runtime bounds check.
struct IndexedByteRange {
  int64_t min;
  int64_t max;
};
} // namespace pto

#endif
