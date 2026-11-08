#ifndef __GLOBAL_ITERATOR_HPP__
#define __GLOBAL_ITERATOR_HPP__

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename glb_tensor, is_tile_data_v tl_tensor>
struct global_iterator {

  using DType = typename glb_tensor::DType;
  global_iterator(DType *data) : data_(data) {}

  /*2D*/
  static_assert(glb_tensor::staticStride[0] == 1 && glb_tensor::staticStride[1] == 1 &&
                "global_iterator can only be use in 2D global_tensor");
  auto operator()(int i, int j) {
    using shape = Shape<1, 1, 1, 1, 1>;
    using stride = Stride<1, 1, glb_tensor::RowStride * glb_tensor::ColStride,
                          glb_tensor::RowStride, glb_tensor::ColStride>;

    using new_tile =
      std::conditional_t<glb_tensor::isRowMajor,
          GlobalTensor<DType, shape, stride, Layout::ND>,
          GlobalTensor<DType, shape, stride, Layout::DN>>;

    // Strides count logical lanes. Packed rows/columns restart at a byte
    // boundary; rounding the whole linear offset would lose that boundary
    // for an odd packed row pitch.
    constexpr size_t Bits = pto::tile_element_bits_v<DType>;
    constexpr size_t RowPitchBytes = (glb_tensor::RowStride * Bits + 7) / 8;
    constexpr size_t ColPitchBytes = (glb_tensor::ColStride * Bits + 7) / 8;
    static_assert(Bits >= 8 ||
                      (glb_tensor::isRowMajor ? tl_tensor::Cols * glb_tensor::ColStride
                                            : tl_tensor::Rows * glb_tensor::RowStride) * Bits % 8 == 0,
                  "packed global_iterator parts must begin at byte boundaries");
    const size_t offset_bytes = glb_tensor::isRowMajor
        ? size_t(i) * tl_tensor::Rows * RowPitchBytes + size_t(j) * tl_tensor::Cols * glb_tensor::ColStride * Bits / 8
        : size_t(i) * tl_tensor::Rows * glb_tensor::RowStride * Bits / 8 + size_t(j) * tl_tensor::Cols * ColPitchBytes;
    new_tile tile(data_ + offset_bytes / sizeof(DType));
    return tile;
  };

private:
  DType *data_;
};

#endif
