/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Integrated from LinxISA/Linx-TileOP-API; see ../../README.md for provenance.

#ifndef PTO_LINX_VENDOR_CPU_UTILS_HPP
#define PTO_LINX_VENDOR_CPU_UTILS_HPP

#include <iomanip>
#include "pto/npu/linx/impl/common/pto_tile.hpp"

namespace pto {
template <typename tile_shape>
void print_tile_Impl(const tile_shape& tile)
{
    static constexpr size_t tile_size = tile_shape::Rows * tile_shape::Cols;
    typename tile_shape::DType d[tile_size];
    std::copy(tile.data(), tile.data() + tile_size, d);

    print_tile_info<tile_shape>();
    std::cout << std::fixed << std::scientific << std::setprecision(4);
    if constexpr (!tile_shape::isBoxedLayout) {
        if constexpr (tile_shape::isRowMajor) {
            for (int i = 0; i < tile_shape::Rows; i++) {
                for (int j = 0; j < tile_shape::Cols; j++) {
                    int offset = i * tile_shape::Cols + j;
                    std::cout << std::setw(8) << static_cast<float>(*(d + offset)) << "\t";
                    if (j == tile_shape::ValidCol - 1 && tile_shape::ValidCol < tile_shape::Cols)
                        std::cout << (i >= tile_shape::ValidRow ? " \t" : "|\t");
                }
                if (i == tile_shape::ValidRow - 1 && tile_shape::ValidRow < tile_shape::Rows) {
                    std::cout << std::endl;
                    for (int k = 0; k < tile_shape::ValidCol; k++)
                        std::cout << std::setw(8) << "-" << "\t";
                }
                std::cout << std::endl;
            }
        }

        if constexpr (!tile_shape::isRowMajor) {
            for (int i = 0; i < tile_shape::Rows; i++) {
                for (int j = 0; j < tile_shape::Cols; j++) {
                    int offset = j * tile_shape::Rows + i;
                    std::cout << std::setw(8) << static_cast<float>(*(d + offset)) << "\t";
                    if (j == tile_shape::ValidCol - 1 && tile_shape::ValidCol < tile_shape::Cols)
                        std::cout << (i >= tile_shape::ValidRow ? " \t" : "|\t");
                }
                if (i == tile_shape::ValidRow - 1 && tile_shape::ValidRow < tile_shape::Rows) {
                    std::cout << std::endl;
                    for (int k = 0; k < tile_shape::ValidCol; k++)
                        std::cout << std::setw(8) << "-" << "\t";
                }
                std::cout << std::endl;
            }
        }

    } else {
        if constexpr (is_Nz_layout<tile_shape>::value) {
            static constexpr int innerTileSize = tile_shape::InnerRows * tile_shape::InnerCols;
            // Fractals total col nums
            static constexpr int nBlockY = tile_shape::Rows / tile_shape::InnerRows;
            // Fractals total row nums
            static constexpr int nBlockX = tile_shape::Cols / tile_shape::InnerCols;
            for (int i = 0; i < tile_shape::Rows; i++) {
                for (int j = 0; j < tile_shape::Cols; j++) {
                    // Fractals coordinate: [blockX, blockY]
                    int blockY = i / tile_shape::InnerRows;
                    int blockX = j / tile_shape::InnerCols;
                    // coordinate in current Fractals: [inX, inY]
                    int inY = i % tile_shape::InnerRows;
                    int inX = j % tile_shape::InnerCols;
                    int offset = (blockX * nBlockY + blockY) * innerTileSize + inY * tile_shape::InnerCols + inX;
                    std::cout << std::setw(8) << static_cast<float>(*(d + offset)) << "\t";
                    if (j == tile_shape::ValidCol - 1 && tile_shape::ValidCol < tile_shape::Cols)
                        std::cout << (i >= tile_shape::ValidRow ? " \t" : "|\t");
                }
                if (i == tile_shape::ValidRow - 1 && tile_shape::ValidRow < tile_shape::Rows) {
                    std::cout << std::endl;
                    for (int k = 0; k < tile_shape::ValidCol; k++)
                        std::cout << std::setw(8) << "-" << "\t";
                }
                std::cout << std::endl;
            }
        }

        if constexpr (is_Zn_layout<tile_shape>::value) {
            static constexpr int innerTileSize = tile_shape::InnerRows * tile_shape::InnerCols;
            // Fractals total col nums
            static constexpr int nBlockY = tile_shape::Rows / tile_shape::InnerRows;
            // Fractals total row nums
            static constexpr int nBlockX = tile_shape::Cols / tile_shape::InnerCols;
            for (int i = 0; i < tile_shape::Rows; i++) {
                for (int j = 0; j < tile_shape::Cols; j++) {
                    // Fractals coordinate: [blockX, blockY]
                    int blockY = i / tile_shape::InnerRows;
                    int blockX = j / tile_shape::InnerCols;
                    // coordinate in current Fractals: [inX, inY]
                    int inY = i % tile_shape::InnerRows;
                    int inX = j % tile_shape::InnerCols;
                    int offset = (blockY * nBlockX + blockX) * innerTileSize + inX * tile_shape::InnerRows + inY;
                    std::cout << std::setw(8) << static_cast<float>(*(d + offset)) << "\t";
                    if (j == tile_shape::ValidCol - 1 && tile_shape::ValidCol < tile_shape::Cols)
                        std::cout << (i >= tile_shape::ValidRow ? " \t" : "|\t");
                }
                if (i == tile_shape::ValidRow - 1 && tile_shape::ValidRow < tile_shape::Rows) {
                    std::cout << std::endl;
#pragma clang loop unroll(full)
                    for (int k = 0; k < tile_shape::ValidCol; k++)
                        std::cout << std::setw(8) << "-" << "\t";
                }
                std::cout << std::endl;
            }
        }
    }
}

} // namespace pto

#endif // PTO_LINX_VENDOR_CPU_UTILS_HPP
