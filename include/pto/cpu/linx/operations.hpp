/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_LINX_OPERATIONS_HPP
#define PTO_CPU_LINX_OPERATIONS_HPP

#include <pto/cpu/linx/numeric.hpp>
#include <pto/cpu/linx/runtime.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <pto/npu/linx/impl/common/pto_local_cube_assembly.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <pto/npu/linx/impl/common/pto_indexed_range.hpp>
#include <pto/npu/linx/impl/common/pto_tile.hpp>
#include <pto/npu/linx/impl/common/pto_tile_region.hpp>

namespace pto::linx_cpu {
template <typename T>
struct AddressSpace6 {
    T data;
};
inline void require(bool condition, const char* message)
{
    if (!condition)
        throw std::invalid_argument(message);
}

template <typename T>
using plain_t = std::remove_cvref_t<T>;
template <typename T>
using dtype_t = typename plain_t<T>::DType;

template <typename T>
constexpr std::size_t capacity()
{
    if constexpr (requires { T::CapacityBytes; })
        return T::CapacityBytes;
    else
        return T::LogicalTileBytes;
}
template <typename T>
std::size_t viewOffset(const T& tile)
{
    if constexpr (requires { tile.GetRangeBase(); }) {
        std::size_t offset = tile.GetRangeBase();
        if constexpr (requires { T::OffsetUnits; })
            offset += T::OffsetUnits;
        return offset * 128;
    } else
        return 0;
}
struct SharedStorage {
    std::vector<std::byte> payload;
    bool published = false;
    bool assemblyOpen = false;
    std::vector<std::pair<std::size_t, std::size_t>> writers;
    CoreContext* context = nullptr;
    CoreContext::SharedExchange* pending = nullptr;
    unsigned participants = 0;
};
inline thread_local std::unordered_map<unsigned long, SharedStorage> sharedStorage;
inline thread_local unsigned long nextSharedHandle = 1;

inline void resolveShared(SharedStorage& storage)
{
    if (!storage.pending)
        return;
    auto& core = *storage.context;
    auto& exchange = *storage.pending;
    std::unique_lock lock(core.mutex);
    core.changed.wait(lock, [&] {
        return exchange.ready || core.error || (core.finished & storage.participants & ~exchange.arrived);
    });
    if (core.error)
        std::rethrow_exception(core.error);
    require(exchange.ready, "Shared producer participant exited before publication");
    require(exchange.result.size() == storage.payload.size(), "Shared participants disagree on capacity");
    storage.payload = exchange.result;
    storage.pending = nullptr;
}

template <typename D>
void prepareDestination(D& dst, bool associated = false, bool active = true)
{
    if constexpr (D::Loc == Location::Shared) {
        bool allocate = !associated;
        if constexpr (is_assemble_v<D>)
            allocate = D::INIT;
        if (allocate || (!active && !sharedStorage.contains(dst.handle()))) {
            dst.handle_ref() = nextSharedHandle++;
            sharedStorage[dst.handle()].payload.resize(capacity<D>());
        } else
            require(sharedStorage.contains(dst.handle()), "Shared assembly has no INIT generation");
        auto& record = sharedStorage[dst.handle()];
        if (!active)
            return;
        resolveShared(record);
        if constexpr (is_assemble_v<D>) {
            if constexpr (D::INIT)
                record.assemblyOpen = true;
            else
                require(record.assemblyOpen, "Shared assembly is already closed");
            constexpr unsigned code = D::WriterSizeCode ? D::WriterSizeCode : D::TilesizeCode;
            const auto first = viewOffset(dst), last = first + (128u << (code - 1));
            require(last <= record.payload.size(), "Shared assembly writer exceeds its parent");
            for (auto [begin, end] : record.writers)
                require(last <= begin || first >= end, "Shared assembly writers overlap");
            record.writers.emplace_back(first, last);
            if constexpr (D::LAST)
                record.assemblyOpen = false;
        }
        record.published = false;
    }
}
template <typename T>
auto bytes(T& tile)
{
    using Byte = std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>;
    if constexpr (T::Loc == Location::Shared) {
        auto it = sharedStorage.find(tile.handle());
        require(it != sharedStorage.end(), "Shared source has no producer generation");
        resolveShared(it->second);
        if constexpr (std::is_const_v<T>)
            require(it->second.published, "Shared source is not published");
        require(viewOffset(tile) < it->second.payload.size(), "Shared view exceeds parent capacity");
        return static_cast<Byte*>(it->second.payload.data()) + viewOffset(tile);
    } else {
        require(viewOffset(tile) < sizeof(const_cast<plain_t<T>&>(tile).data()), "Local view exceeds parent capacity");
        return reinterpret_cast<Byte*>(&const_cast<plain_t<T>&>(tile).data()) + viewOffset(tile);
    }
}
template <typename T>
void shape(const T& tile)
{
    require(
        tile.GetValidRow() >= 0 && tile.GetValidRow() <= T::Rows && tile.GetValidCol() >= 0 &&
            tile.GetValidCol() <= T::Cols,
        "Linx valid shape exceeds physical tile shape");
}
template <typename T>
void checkAccess(const T& tile, std::size_t offset, std::size_t length)
{
    std::size_t total;
    if constexpr (T::Loc == Location::Shared)
        total = sharedStorage.at(tile.handle()).payload.size();
    else
        total = sizeof(const_cast<plain_t<T>&>(tile).data());
    const auto base = viewOffset(tile);
    require(
        base <= total && offset <= total - base && length <= total - base - offset,
        "Linx Tile access exceeds its parent allocation");
    if constexpr (requires { T::SubviewSizeCode; })
        require(offset + length <= (128u << (T::SubviewSizeCode - 1)), "Linx source exceeds its subview");
    if constexpr (is_assemble_v<T>) {
        constexpr unsigned writer = T::WriterSizeCode ? T::WriterSizeCode : T::TilesizeCode;
        require(offset + length <= (128u << (writer - 1)), "Linx writer exceeds its assembly fragment");
    }
}

template <typename T>
std::size_t elementIndex(std::size_t row, std::size_t col)
{
    using E = dtype_t<T>;
    constexpr int width = bits<E>;
    if constexpr (T::BFractal == BLayout::CubeM32 || T::BFractal == BLayout::CubeM16) {
        constexpr int rows = T::BFractal == BLayout::CubeM32 ? 32 : 16;
        constexpr int columns = 1024 / (rows * width);
        if constexpr (width == 64 && rows == 32)
            return col * 32 + row;
        else {
            auto inner = col % columns;
            if constexpr (width == 4 && rows == 16) {
                if (inner >= 4 && inner < 8)
                    inner += 4;
                else if (inner >= 8 && inner < 12)
                    inner -= 4;
            }
            return col / columns * rows * columns + row * columns + inner;
        }
    } else if constexpr (T::BFractal == BLayout::CubeN8) {
        constexpr int rows = 1024 / (8 * width);
        constexpr int repeats = (T::Rows + rows - 1) / rows;
        return (col / 8 * repeats + row / rows) * rows * 8 + col % 8 * rows + row % rows;
    } else if constexpr (T::SFractal != SLayout::NoneBox) {
        constexpr int ir = T::InnerRows, ic = T::InnerCols;
        auto outer = T::BFractal == BLayout::RowMajor ? row / ir * (T::Cols / ic) + col / ic :
                                                        col / ic * (T::Rows / ir) + row / ir;
        auto inner = T::SFractal == SLayout::RowMajor ? row % ir * ic + col % ic : col % ic * ir + row % ir;
        return outer * ir * ic + inner;
    } else
        return T::BFractal == BLayout::RowMajor ? row * T::Cols + col : col * T::Rows + row;
}
template <typename T>
uint64_t readBits(const T& tile, std::size_t row, std::size_t col)
{
    using E = dtype_t<T>;
    const auto* ptr = bytes(tile);
    uint64_t raw = 0;
    if constexpr (T::BFractal == BLayout::CubeM32 && bits<E> == 64) {
        checkAccess(tile, col * 256 + 128 + row * 4, 4);
        uint32_t low, high;
        std::memcpy(&low, ptr + col * 256 + row * 4, 4);
        std::memcpy(&high, ptr + col * 256 + 128 + row * 4, 4);
        return uint64_t(high) << 32 | low;
    } else {
        const auto index = elementIndex<T>(row, col);
        checkAccess(tile, index * bits<E> / 8, (bits<E> + 7) / 8);
        if constexpr (bits<E> == 4)
            return (std::to_integer<unsigned>(ptr[index / 2]) >> ((index & 1) * 4)) & 15;
        else {
            std::memcpy(&raw, ptr + index * sizeof(E), sizeof(E));
            return raw;
        }
    }
}
template <typename T>
void writeBits(T& tile, std::size_t row, std::size_t col, uint64_t raw)
{
    using E = dtype_t<T>;
    auto* ptr = bytes(tile);
    if constexpr (T::BFractal == BLayout::CubeM32 && bits<E> == 64) {
        checkAccess(tile, col * 256 + 128 + row * 4, 4);
        uint32_t low = raw, high = raw >> 32;
        std::memcpy(ptr + col * 256 + row * 4, &low, 4);
        std::memcpy(ptr + col * 256 + 128 + row * 4, &high, 4);
    } else {
        const auto index = elementIndex<T>(row, col);
        checkAccess(tile, index * bits<E> / 8, (bits<E> + 7) / 8);
        if constexpr (bits<E> == 4) {
            const auto shift = (index & 1) * 4;
            ptr[index / 2] =
                std::byte((std::to_integer<unsigned>(ptr[index / 2]) & ~(15 << shift)) | ((raw & 15) << shift));
        } else
            std::memcpy(ptr + index * sizeof(E), &raw, sizeof(E));
    }
}
template <typename T>
auto read(const T& tile, std::size_t row, std::size_t col)
{
    return fromBits<dtype_t<T>>(readBits(tile, row, col));
}
template <typename D>
void publishShared(D& dst, unsigned mask, bool publish);

template <typename D>
bool skipSharedProducer(D& dst, unsigned mask, bool publish = true, bool associated = false)
{
    if (mask == 0)
        return true;
    if constexpr (D::Loc == Location::Shared) {
        if (currentCore && !(mask & (8u >> threadIndex()))) {
            prepareDestination(dst, associated, false);
            publishShared(dst, mask, publish);
            return true;
        }
    }
    return false;
}

template <typename T, typename F>
void produce(
    T& dst, std::size_t columns, std::size_t rows, F fn, unsigned peMask = 15, bool publish = true,
    bool associated = false)
{
    if (peMask == 0)
        return;
    if (skipSharedProducer(dst, peMask, publish, associated))
        return;
    require(columns <= T::Cols && rows <= T::Rows, "Linx operation shape exceeds destination");
    // Every instruction consumes its complete input generation before publishing output.
    std::vector<uint64_t> result(rows * columns);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < columns; ++c) {
            bool selected = true;
            if constexpr (T::Loc == Location::Shared) {
                if (currentCore) {
                    if (std::popcount(peMask) == 1)
                        selected = (peMask & (8u >> threadIndex())) != 0;
                    else if constexpr (is_assemble_v<T>)
                        selected = threadIndex() == (viewOffset(dst) / (128u << (T::WriterSizeCode - 1))) % 4;
                    else
                        selected = (peMask & (8u >> threadIndex())) &&
                                   (elementIndex<T>(r, c) * bits<dtype_t<T>> / 8 * 4 / capacity<T>() == threadIndex());
                }
            }
            if (selected)
                result[r * columns + c] = fn(r, c);
        }
    prepareDestination(dst, associated);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < columns; ++c)
            writeBits(dst, r, c, result[r * columns + c]);
    if constexpr (T::Loc == Location::Shared)
        publishShared(dst, peMask, publish);
}
template <int Op, typename T>
T calculate(T a, T b)
{
    if constexpr (integral<T>) {
        const uint64_t x = scalarBits(a), y = scalarBits(b);
        if constexpr (Op == 0)
            return fromBits<T>(x + y);
        if constexpr (Op == 1)
            return fromBits<T>(x - y);
        if constexpr (Op == 2)
            return fromBits<T>(x * y);
        if constexpr (Op == 6)
            return fromBits<T>(x & y);
        if constexpr (Op == 7)
            return fromBits<T>(x | y);
        if constexpr (Op == 8)
            return fromBits<T>(x ^ y);
        if constexpr (Op == 9)
            return fromBits<T>(x << (y % bits<T>));
        if constexpr (Op == 10) {
            if constexpr (code<T> < 24)
                return encode<T>(static_cast<int64_t>(number(a)) >> (y % bits<T>));
            else
                return fromBits<T>(x >> (y % bits<T>));
        }
        if constexpr (Op == 3 || Op == 4) {
            require(y != 0, "Linx integer division by zero");
            if constexpr (code<T> < 24) {
                auto lhs = static_cast<int64_t>(number(a)), rhs = static_cast<int64_t>(number(b));
                if (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)
                    return Op == 3 ? a : fromBits<T>(0);
                if constexpr (Op == 3)
                    return fromBits<T>(uint64_t(lhs / rhs));
                auto remainder = lhs % rhs;
                if (remainder != 0 && (remainder < 0) != (rhs < 0))
                    remainder += rhs;
                return fromBits<T>(uint64_t(remainder));
            } else
                return fromBits<T>(Op == 3 ? x / y : x % y);
        }
    }
    if constexpr (std::is_same_v<T, double>) {
        if constexpr (Op == 0)
            return a + b;
        if constexpr (Op == 1)
            return a - b;
        if constexpr (Op == 2)
            return a * b;
        if constexpr (Op == 3)
            return a / b;
    }
    const long double x = number(a), y = number(b);
    if constexpr (Op == 0)
        return encode<T>(x + y);
    if constexpr (Op == 1)
        return encode<T>(x - y);
    if constexpr (Op == 2)
        return encode<T>(x * y);
    if constexpr (Op == 3)
        return encode<T>(x / y);
    if constexpr (Op == 4)
        return encode<T>(std::fmod(x, y));
    if constexpr (Op == 11 || Op == 12)
        return fromBits<T>(minMax(code<T>, scalarBits(a), scalarBits(b), Op == 11));
    if constexpr (Op == 29) {
        // TEXPDIF is a typed SUB followed by EXP, not a fused exp(a-b).
        const auto difference = encode<T>(x - y);
        return encode<T>(std::exp(static_cast<float>(number(difference))));
    }
    throw std::invalid_argument("Linx binary operation does not support this dtype");
}
template <int Op, typename D, typename A, typename B>
void binary(D& dst, A& a, B& b, unsigned columns = 0, unsigned rows = 0)
{
    shape(a);
    shape(b);
    columns = columns ? columns : a.GetValidCol();
    rows = rows ? rows : a.GetValidRow();
    produce(dst, columns, rows, [&](auto r, auto c) {
        if constexpr (Op == 29) {
            // Widening forms subtract in FP32; same-type forms round the
            // intermediate subtraction in FP16/BF16/FP32 before EXP.
            const auto difference = encode<dtype_t<D>>(number(read(a, r, c)) - number(read(b, r, c)));
            return scalarBits(encode<dtype_t<D>>(std::exp(static_cast<float>(number(difference)))));
        } else
            return scalarBits(calculate<Op>(read(a, r, c), fromBits<dtype_t<A>>(readBits(b, r, c))));
    });
}
template <int Op, typename D, typename S>
void unary(D& dst, S& src)
{
    shape(src);
    produce(dst, src.GetValidCol(), src.GetValidRow(), [&](auto r, auto c) {
        auto raw = readBits(src, r, c);
        if constexpr (Op == 17 && integral<dtype_t<S>>)
            return uint64_t{0} - raw;
        if constexpr (Op == 16)
            return ~raw;
        if constexpr ((Op == 15 || Op == 17) && code<dtype_t<S>> <= 5) {
            const auto f = binaryFormat(code<dtype_t<S>>);
            const uint64_t sign = uint64_t{1} << (f.exponent + f.fraction + f.shift);
            return Op == 15 ? raw & ~sign : raw ^ sign;
        }
        const long double x = number(fromBits<dtype_t<S>>(raw));
        long double value;
        if constexpr (Op == 15)
            value = std::fabs(x);
        else if constexpr (Op == 17)
            value = -x;
        else if constexpr (Op == 18)
            value = std::exp(x);
        else if constexpr (Op == 19) {
            // The native model rounds log2 and the ln(2) multiply separately.
            // Narrow floating formats use FP32 before the final conversion.
            if constexpr (std::is_same_v<dtype_t<S>, double>) {
                const double log2 = std::log2(static_cast<double>(x));
                value = log2 * 0x1.62e42fefa39efp-1;
            } else if constexpr (!integral<dtype_t<S>>) {
                const float log2 = std::log2(static_cast<float>(x));
                value = log2 * 0x1.62e430p-1f;
            } else
                value = std::log(x);
        } else if constexpr (Op == 20)
            value = 1 / x;
        else if constexpr (Op == 21)
            value = std::sqrt(x);
        else if constexpr (Op == 22) {
            // Pinned Linx model contract: typed SQRT followed by typed DIV,
            // including FP16/BF16. This differs from the pinned specification's
            // single-round reference; see the backend's simulation contract.
            if constexpr (!integral<dtype_t<S>> && code<dtype_t<S>> <= 5)
                value = number(calculate<3>(encode<dtype_t<S>>(1), encode<dtype_t<S>>(std::sqrt(x))));
            else
                value = 1 / std::sqrt(x);
        } else if constexpr (Op == 23)
            value = std::isnan(x) ? x : x > 0 ? x : 0.L;
        return scalarBits(encode<dtype_t<D>>(value));
    });
}
template <int Op, typename D, typename S, typename V>
void scalar(D& dst, S& src, V value)
{
    shape(src);
    produce(dst, src.GetValidCol(), src.GetValidRow(), [&](auto r, auto c) {
        return scalarBits(calculate<Op>(read(src, r, c), value));
    });
}
template <typename D, typename V>
void fill(D& dst, V value)
{
    shape(dst);
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto, auto) { return scalarBits(value); });
}
template <typename D, typename S>
void copy(D& dst, const S& src, unsigned mask = 15, bool publish = true)
{
    if (mask == 0)
        return;
    if (skipSharedProducer(dst, mask, publish))
        return;
    shape(src);
    produce(
        dst, src.GetValidCol(), src.GetValidRow(), [&](auto r, auto c) { return readBits(src, r, c); }, mask, publish);
}
template <int Mask = 15, int Pad = 0, bool Associated = false, typename D, typename G>
void load(D& dst, const G& src)
{
    if (skipSharedProducer(dst, Mask, true, Associated))
        return;
    shape(dst);
    produce(
        dst, dst.GetValidCol(), dst.GetValidRow(),
        [&](auto r, auto c) {
            uint64_t value = 0;
            auto* p = reinterpret_cast<const std::byte*>(src.data()) + r * src.GetStrideBytes(3);
            if constexpr (bits<dtype_t<G>> == 4)
                value = (std::to_integer<unsigned>(p[c / 2]) >> ((c % 2) * 4)) & 15;
            else
                std::memcpy(&value, p + c * sizeof(dtype_t<G>), sizeof(dtype_t<G>));
            return value;
        },
        Mask, true, Associated);
    // Null leaves the physical tail untouched; the other encodings initialize
    // only the cells belonging to the persistent CUBE rectangle.
    if constexpr (D::IsCubeLayout && Pad != 3) {
        const auto value = Pad == 0 ? uint64_t{0} :
                                      encodeNumber(
                                          code<dtype_t<D>>,
                                          Pad == 1 ? std::numeric_limits<long double>::infinity() :
                                                     -std::numeric_limits<long double>::infinity(),
                                          0, true);
        constexpr int physicalRows = [] {
            if constexpr (requires { D::PhysicalRows; })
                return D::PhysicalRows;
            else
                return D::Rows;
        }();
        constexpr int physicalCols = [] {
            if constexpr (requires { D::PhysicalCols; })
                return D::PhysicalCols;
            else
                return D::Cols;
        }();
        for (int r = 0; r < physicalRows; ++r)
            for (int c = 0; c < physicalCols; ++c)
                if (r >= dst.GetValidRow() || c >= dst.GetValidCol())
                    writeBits(dst, r, c, value);
    }
}
template <int Mask = 15, typename G, typename S>
void store(G& dst, const S& src)
{
    if (!(Mask & (8u >> threadIndex())))
        return;
    shape(src);
    for (int r = 0; r < src.GetValidRow(); ++r)
        for (int c = 0; c < src.GetValidCol(); ++c) {
            auto value = readBits(src, r, c);
            auto* p = reinterpret_cast<std::byte*>(dst.data()) + r * dst.GetStrideBytes(3);
            if constexpr (bits<dtype_t<S>> == 4) {
                unsigned shift = (c % 2) * 4;
                p[c / 2] = std::byte((std::to_integer<unsigned>(p[c / 2]) & ~(15u << shift)) | ((value & 15) << shift));
            } else
                std::memcpy(p + c * sizeof(dtype_t<S>), &value, sizeof(dtype_t<S>));
        }
}
template <int RMode, typename D, typename S>
void convert(D& dst, const S& src)
{
    produce(dst, src.GetValidCol(), src.GetValidRow(), [&](auto r, auto c) {
        return convertRaw(code<dtype_t<D>>, code<dtype_t<S>>, readBits(src, r, c), RMode, false);
    });
}
template <int Descending, typename D, typename V>
void sequence(D& dst, V value)
{
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto, auto c) {
        return scalarBits(encode<dtype_t<D>>(number(value) + (Descending ? -1.L : 1.L) * c));
    });
}
template <int RowStep, int ColStep, typename D, typename V>
void sequence2d(D& dst, V value)
{
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
        return scalarBits(encode<dtype_t<D>>(number(value) + RowStep * int64_t(r) + ColStep * int64_t(c)));
    });
}
template <typename D>
void triangle(D& dst)
{
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
        return scalarBits(encode<dtype_t<D>>(c <= r ? 1 : 0));
    });
}
template <typename D, typename A, typename B, typename C>
void fma(D& dst, A& a, B& b, C& c)
{
    produce(dst, a.GetValidCol(), a.GetValidRow(), [&](auto r, auto col) {
        if constexpr (integral<dtype_t<A>>)
            return readBits(a, r, col) * readBits(b, r, col) + readBits(c, r, col);
        else if constexpr (std::is_same_v<dtype_t<A>, float> || std::is_same_v<dtype_t<A>, double>)
            return scalarBits(std::fma(read(a, r, col), read(b, r, col), read(c, r, col)));
        else
            return scalarBits(encode<dtype_t<D>>(
                std::fma(number(read(a, r, col)), number(read(b, r, col)), number(read(c, r, col)))));
    });
}
template <int Op, typename D, typename S>
void reduce(D& dst, S& src)
{
    constexpr bool rowWise = Op < 80;
    constexpr int kind = rowWise ? Op - 64 : Op - 80;
    const int count = rowWise ? src.GetValidCol() : src.GetValidRow();
    produce(dst, rowWise ? 1 : src.GetValidCol(), rowWise ? src.GetValidRow() : 1, [&](auto r, auto c) {
        require(count > 0, "empty reduction");
        auto acc = (kind == 0) ? encode<dtype_t<S>>(0) : (kind == 3) ? encode<dtype_t<S>>(1) : read(src, r, c);
        unsigned index = 0;
        for (int i = 0; i < count; ++i) {
            auto x = read(src, rowWise ? r : i, rowWise ? i : c);
            if constexpr (kind == 0)
                acc = calculate<0>(acc, x);
            else if constexpr (kind == 3)
                acc = calculate<2>(acc, x);
            else if constexpr (kind == 1 || kind == 12 || kind == 2 || kind == 13) {
                auto next = calculate < (kind == 1 || kind == 12) ? 11 : 12 > (acc, x);
                if (scalarBits(next) == scalarBits(x) && scalarBits(next) != scalarBits(acc))
                    index = i;
                acc = next;
            }
        }
        if constexpr (kind == 12 || kind == 13)
            return uint64_t(index);
        else
            return scalarBits(acc);
    });
}
template <bool RowWise, int ByteOffset = 0, typename D, typename S>
void expand(D& dst, S& src)
{
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
        return readBits(src, RowWise ? r : 0, RowWise ? ByteOffset / sizeof(dtype_t<S>) : c);
    });
}
template <int Op, bool RowWise, int ByteOffset = 0, typename D, typename A, typename B>
void broadcast(D& dst, A& a, B& b)
{
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
        return scalarBits(
            calculate<Op>(read(a, r, c), read(b, RowWise ? r : 0, RowWise ? ByteOffset / sizeof(dtype_t<B>) : c)));
    });
}
struct NoOperand {};
inline constexpr NoOperand noOperand{};
inline constexpr IndexedByteRange unboundedRange{
    std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};

template <CmpMode Mode, typename T>
bool compareValue(T a, T b)
{
    const auto x = number(a), y = number(b);
    if constexpr (Mode == CmpMode::EQ)
        return x == y;
    if constexpr (Mode == CmpMode::NE)
        return x != y;
    if constexpr (Mode == CmpMode::LT)
        return x < y;
    if constexpr (Mode == CmpMode::GT)
        return x > y;
    if constexpr (Mode == CmpMode::LE)
        return x <= y;
    if constexpr (Mode == CmpMode::GE)
        return x >= y;
}
template <typename Basis>
std::size_t predicateIndex(std::size_t r, std::size_t c)
{
    constexpr int rows = Basis::BFractal == BLayout::CubeM16 ? 16 : 32;
    constexpr int cols = 128 / rows;
    return c / cols * 128 + r * cols + c % cols;
}
template <typename Basis, typename M>
bool maskBit(const M& mask, std::size_t r, std::size_t c)
{
    const auto* p = bytes(mask);
    if constexpr (Basis::BFractal == BLayout::RowMajor) {
        const auto i = r * Basis::Cols + c;
        return (std::to_integer<unsigned>(p[i / 8]) >> (i % 8)) & 1;
    } else
        return std::to_integer<unsigned>(p[predicateIndex<Basis>(r, c)]) != 0;
}
template <typename Basis, typename D, typename Fn>
void predicate(D& dst, const Basis& src, Fn fn)
{
    const std::size_t extent = [] {
        if constexpr (is_assemble_v<D>) {
            constexpr unsigned writer = D::WriterSizeCode ? D::WriterSizeCode : D::TilesizeCode;
            return std::size_t(128) << (writer - 1);
        } else
            return capacity<D>();
    }();
    std::vector<std::byte> result(extent);
    for (int r = 0; r < src.GetValidRow(); ++r)
        for (int c = 0; c < src.GetValidCol(); ++c) {
            if constexpr (Basis::BFractal == BLayout::RowMajor) {
                const auto i = r * Basis::Cols + c;
                require(i / 8 < result.size(), "predicate exceeds destination capacity");
                if (fn(r, c))
                    result[i / 8] |= std::byte(1 << (i % 8));
            } else {
                const auto i = predicateIndex<Basis>(r, c);
                require(i < result.size(), "predicate CELL exceeds destination capacity");
                result[i] = fn(r, c) ? std::byte{1} : std::byte{};
            }
        }
    prepareDestination(dst);
    checkAccess(dst, 0, result.size());
    std::memcpy(bytes(dst), result.data(), result.size());
    if constexpr (D::Loc == Location::Shared)
        publishShared(dst, 15, true);
}
template <CmpMode Mode, typename D, typename A, typename B>
void compare(D& dst, A& a, B& b)
{
    predicate(dst, a, [&](auto r, auto c) {
        return compareValue<Mode>(read(a, r, c), fromBits<dtype_t<A>>(readBits(b, r, c)));
    });
}
template <CmpMode Mode, typename D, typename S, typename V>
void compareScalar(D& dst, S& src, V value)
{
    predicate(dst, src, [&](auto r, auto c) { return compareValue<Mode>(read(src, r, c), value); });
}
template <CmpMode Mode, bool High, typename S, typename V>
void assignCompareGpr(uint64_t& result, S& src, V value)
{
    constexpr int width = S::BFractal == BLayout::CubeM16 ? 16 : 32;
    constexpr int fields = width == 32 ? 2 : (bits<dtype_t<S>> == 32 ? 2 : 4);
    result = 0;
    for (int c = 0; c < fields; ++c)
        for (int r = 0; r < src.GetValidRow(); ++r) {
            const int col = c + (High ? fields : 0);
            if (col < src.GetValidCol() && compareValue<Mode>(read(src, r, col), value))
                result |= uint64_t{1} << (c * width + r);
        }
}
template <typename D, typename M, typename A, typename B>
void select(D& dst, M& mask, A& a, B& b)
{
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
        return maskBit<D>(mask, r, c) ? readBits(a, r, c) : readBits(b, r, c);
    });
}
template <typename D, typename M, typename A, typename V>
void selectScalar(D& dst, M& mask, A& a, V value)
{
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
        return maskBit<D>(mask, r, c) ? readBits(a, r, c) : scalarBits(value);
    });
}
template <typename G>
void prefetch(const G&, unsigned, unsigned)
{}

template <typename D, typename S, typename I>
void gather(D& dst, S& src, I& off)
{
    produce(dst, off.GetValidCol(), off.GetValidRow(), [&](auto r, auto c) {
        auto i = static_cast<uint64_t>(number(read(off, r, c)));
        require(
            i < std::size_t(src.GetValidRow()) && c < std::size_t(src.GetValidCol()),
            "TGATHER source row exceeds valid shape");
        return readBits(src, i, c);
    });
}
template <typename D, typename S, typename I>
void scatter(D& dst, S& src, I& off)
{
    std::vector<std::pair<uint64_t, uint64_t>> writes;
    for (int r = 0; r < off.GetValidRow(); ++r)
        for (int c = 0; c < off.GetValidCol(); ++c) {
            auto i = static_cast<uint64_t>(number(read(off, r, c)));
            require(
                i < std::size_t(dst.GetValidRow()) && c < dst.GetValidCol(),
                "TSCATTER destination row exceeds valid shape");
            writes.emplace_back(i * dst.GetValidCol() + c, readBits(src, r, c));
        }
    for (int r = 0; r < dst.GetValidRow(); ++r)
        for (int c = 0; c < dst.GetValidCol(); ++c)
            writeBits(dst, r, c, 0);
    for (auto [i, v] : writes)
        writeBits(dst, i / dst.GetValidCol(), i % dst.GetValidCol(), v);
}
inline std::uintptr_t indexedAddress(const void* base, int64_t displacement, IndexedByteRange range, std::size_t width)
{
    require(
        range.min <= range.max && displacement >= range.min && displacement <= range.max,
        "indexed byte displacement exceeds certified range");
    auto address = reinterpret_cast<std::uintptr_t>(base);
    if (displacement < 0) {
        const uint64_t magnitude = 0 - uint64_t(displacement);
        require(address >= magnitude, "indexed address underflow");
        address -= magnitude;
    } else {
        require(address <= std::numeric_limits<std::uintptr_t>::max() - displacement, "indexed address overflow");
        address += displacement;
    }
    require(address <= std::numeric_limits<std::uintptr_t>::max() - (width - 1), "indexed access wraps address space");
    return address;
}
template <typename M>
bool indexedMaskActive(const M& mask, std::size_t row, std::size_t col)
{
    // Indexed TLSU masks contain canonical U8 lanes, not RowMajor TCMP bits.
    const std::size_t index =
        plain_t<M>::BFractal == BLayout::RowMajor ? row * plain_t<M>::Cols + col : predicateIndex<M>(row, col);
    checkAccess(mask, index, 1);
    const auto value = std::to_integer<unsigned>(bytes(mask)[index]);
    require(value <= 1, "indexed mask byte must be 0 or 1");
    return value != 0;
}
template <int Pad, typename D, typename G, typename I, typename M>
void memoryGather(D& dst, const G& src, const I& offset, const M& mask, IndexedByteRange range)
{
    produce(dst, offset.GetValidCol(), offset.GetValidRow(), [&](auto r, auto c) {
        bool active = true;
        if constexpr (!std::is_same_v<M, NoOperand>)
            active = indexedMaskActive(mask, r, c);
        if (!active) {
            if constexpr (Pad == 1 || Pad == 2)
                return encodeNumber(
                    code<dtype_t<D>>,
                    Pad == 1 ? std::numeric_limits<long double>::infinity() :
                               -std::numeric_limits<long double>::infinity(),
                    0, true);
            else
                return uint64_t{0};
        }
        const int64_t off = static_cast<int64_t>(number(read(offset, r, c)));
        const auto p = indexedAddress(src.data(), off, range, sizeof(dtype_t<D>));
        uint64_t value = 0;
        std::memcpy(&value, reinterpret_cast<const void*>(p), sizeof(dtype_t<D>));
        return value;
    });
}
template <typename G, typename S, typename I, typename M>
void memoryScatter(G& dst, const S& src, const I& offset, const M& mask, IndexedByteRange range)
{
    std::vector<std::pair<std::uintptr_t, uint64_t>> writes;
    for (int r = 0; r < offset.GetValidRow(); ++r)
        for (int c = 0; c < offset.GetValidCol(); ++c) {
            if constexpr (!std::is_same_v<M, NoOperand>)
                if (!indexedMaskActive(mask, r, c))
                    continue;
            const int64_t off = static_cast<int64_t>(number(read(offset, r, c)));
            writes.emplace_back(indexedAddress(dst.data(), off, range, sizeof(dtype_t<S>)), readBits(src, r, c));
        }
    for (auto [p, value] : writes)
        std::memcpy(reinterpret_cast<void*>(p), &value, sizeof(dtype_t<S>));
}
template <int Op, typename D, typename A, typename B>
void broadcastOpcode(D& dst, A& a, B& b)
{
    constexpr bool row = Op < 80;
    constexpr int kind = Op - (row ? 69 : 85);
    constexpr int arithmetic = kind < 4 ? kind : kind == 4 ? 11 : kind == 5 ? 12 : 29;
    broadcast<arithmetic, row>(dst, a, b);
}

template <int Mask, bool Publish, typename D, typename S>
void sharedCopy(D& dst, const S& src)
{
    copy(dst, src, Mask, Publish);
}
template <int Mask, typename D, typename S>
void sharedExtract(D& dst, const S& src)
{
    if (Mask & (8u >> threadIndex()))
        copy(dst, src);
}

template <FixpAttr Attr>
uint64_t matrixPostprocess(uint64_t raw, int accumulatorType, int destinationType, uint64_t quant, uint64_t relu)
{
    constexpr int mode = static_cast<int>(Attr.PreQuant);
    if constexpr (mode == 0 && Attr.Relu == FixpReluMode::None)
        return raw;
    auto value = decodeNumber(accumulatorType, raw);
    if constexpr (mode == 12 || mode == 13) {
        int32_t x = std::bit_cast<int32_t>(uint32_t(raw));
        const int shift = ((quant >> 32) & 15) + 1;
        return encodeNumber(18, x >> shift, 1, true);
    }
    constexpr bool hasParameter = mode != 0 && mode != 1 && mode != 16;
    long double scale = hasParameter ? decodeNumber(1, uint32_t(quant) & 0xffffe000u) : 1.L;
    if (std::signbit(value)) {
        if constexpr (Attr.Relu == FixpReluMode::Relu)
            scale = 0;
        else if constexpr (Attr.Relu == FixpReluMode::LRelu || Attr.Relu == FixpReluMode::PRelu)
            scale = decodeNumber(1, (relu & 0x7ffff) << 13);
    }
    // FPATR classifies the accumulator before scaling or the intermediate
    // integer clamp. Negative infinity times a zero activation is the one
    // special-value case that proceeds as a finite zero.
    if (!std::isfinite(value) && !(std::isinf(value) && std::signbit(value) && scale == 0)) {
        if (integerCode(destinationType))
            return destinationType < 24 ? uint64_t{1} << (numericBits(destinationType) - 1) : 0;
        return encodeNumber(destinationType, value, 1, false);
    }
    if (scale == 0 && std::isinf(value))
        value = 0;
    else
        value *= scale;
    constexpr int width = (mode == 17 || mode == 18)                           ? 5 :
                          (mode == 2 || mode == 3 || mode == 23 || mode == 24) ? 9 :
                          (mode == 19 || mode == 20)                           ? 17 :
                                                                                 0;
    if constexpr (width != 0) {
        value = std::clamp(roundIntegral(value, 1), -std::ldexp(1.L, width - 1), std::ldexp(1.L, width - 1) - 1);
        uint64_t rawOffset = (quant >> 37) & ((1u << width) - 1);
        int64_t offset = (rawOffset & (1u << (width - 1))) ? int64_t(rawOffset) - (1 << width) : int64_t(rawOffset);
        value += offset;
    }
    if (integerCode(destinationType) && !std::isfinite(value))
        return destinationType < 24 ? uint64_t{1} << (numericBits(destinationType) - 1) : 0;
    return encodeNumber(destinationType, value, mode == 25 || mode == 28 ? 7 : 1, false);
}
template <typename S>
long double matrixScale(const S& scale, std::size_t row, std::size_t group)
{
    return number(read(scale, row, group));
}
// Matrix bundles snapshot all operands and all auxiliary outputs before any
// publication. This also permits the native ACC in-place D == C contract.
template <
    FixpAttr Attr, bool Accumulate, bool Bias, int ScaleMask, int SrcMask, int OutMask, typename D, typename A,
    typename B, typename C, typename SA, typename SB, typename CS, typename RI, typename Q, typename P, typename RO,
    typename GO>
void matrix(
    D& dst, const A& a, const B& b, const C& extra, const SA& scaleA, const SB& scaleB, const CS& cscale,
    const RI& rowIn, const Q& quantTile, const P& reluTile, RO& rowOut, GO& groupOut, uint64_t quantGpr,
    uint64_t reluGpr, std::size_t M, std::size_t N, std::size_t K)
{
    constexpr bool sharedA = A::Loc == Location::Shared, sharedB = B::Loc == Location::Shared;
    constexpr bool floating = !integral<dtype_t<A>>;
    constexpr int accumulatorType =
        floating ? __type_fp32 : (code<dtype_t<A>> >= 24 && code<dtype_t<B>> >= 24 ? __type_uint32 : __type_int32);
    const std::size_t shard = (sharedA || sharedB) ? threadIndex() * (M <= 64 ? 16 : 32) : 0;
    const std::size_t rows = shard < M ? std::min<std::size_t>(dst.GetValidRow(), M - shard) : 0;
    require(N <= std::size_t(dst.GetValidCol()), "matrix N exceeds destination valid columns");
    std::vector<uint64_t> intermediates(rows * N), output(rows * N), rowResults(rows), groupResults;
    constexpr unsigned groupN = Attr.GroupNCode == 0 ? 0 : Attr.GroupNCode == 1 ? 8 : (Attr.GroupNCode - 1) * 16;
    constexpr int groups = Attr.GroupMaxEn ? (D::ValidCol + groupN - 1) / groupN : 0;
    if constexpr (OutMask & 2)
        groupResults.resize(rows * groups);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t n = 0; n < N; ++n) {
            uint64_t acc = 0;
            if constexpr (Accumulate) {
                acc = readBits(extra, r, n);
                if constexpr (Attr.CScaleEn)
                    acc = encodeNumber(
                        accumulatorType,
                        std::ldexp(decodeNumber(accumulatorType, acc), -int(readBits(cscale, r, 0) & 255)), 1);
            }
            for (std::size_t k = 0; k < K; ++k) {
                auto av = [&] {
                    if constexpr (sharedA && Attr.TransA)
                        return read(a, k, r + shard);
                    else
                        return read(a, r + (sharedA ? shard : 0), k);
                }();
                auto bv = [&] {
                    if constexpr (sharedB && !Attr.TransB)
                        return read(b, n, k);
                    else
                        return read(b, k, n);
                }();
                if constexpr (floating) {
                    long double x = number(av), y = number(bv);
                    if constexpr (ScaleMask & 1) {
                        constexpr auto group = code<dtype_t<A>> == __type_fp4_hif4x2 ? 64 : 32;
                        if constexpr (code<dtype_t<A>> == __type_fp4_hif4x2) {
                            auto word = readBits(scaleA, r + (sharedA ? shard : 0), k / group);
                            x *= std::ldexp(
                                decodeNumber(__type_fp8_e6m2, word & 255),
                                ((word >> (8 + k % 64 / 8)) & 1) + ((word >> (16 + k % 64 / 4)) & 1));
                        } else
                            x *= matrixScale(scaleA, r + (sharedA ? shard : 0), k / group);
                    }
                    if constexpr (ScaleMask & 2) {
                        constexpr auto group = code<dtype_t<B>> == __type_fp4_hif4x2 ? 64 : 32;
                        auto scale = [&] {
                            if constexpr (code<dtype_t<B>> == __type_fp4_hif4x2) {
                                auto word = (SB::Loc == Location::Shared || SB::BFractal == BLayout::CubeM32) ?
                                                readBits(scaleB, n, k / group) :
                                                readBits(scaleB, k / group, n);
                                return std::ldexp(
                                    decodeNumber(__type_fp8_e6m2, word & 255),
                                    ((word >> (8 + k % 64 / 8)) & 1) + ((word >> (16 + k % 64 / 4)) & 1));
                            } else if constexpr (SB::Loc == Location::Shared || SB::BFractal == BLayout::CubeM32)
                                return matrixScale(scaleB, n, k / group);
                            else
                                return matrixScale(scaleB, k / group, n);
                        }();
                        y *= scale;
                    }
                    auto product = encodeNumber(__type_fp32, x * y, 1);
                    acc = encodeNumber(
                        __type_fp32, decodeNumber(__type_fp32, acc) + decodeNumber(__type_fp32, product), 1);
                } else {
                    auto x = static_cast<int64_t>(number(av)), y = static_cast<int64_t>(number(bv));
                    acc = uint32_t(acc + uint64_t(x) * uint64_t(y));
                }
            }
            if constexpr (Bias) {
                if constexpr (floating)
                    acc = encodeNumber(
                        accumulatorType, decodeNumber(accumulatorType, acc) + number(read(extra, 0, n)), 1);
                else
                    acc = uint32_t(acc + readBits(extra, 0, n));
            }
            intermediates[r * N + n] = acc;
            uint64_t q = quantGpr, p = reluGpr;
            if constexpr (SrcMask & 2)
                q = readBits(quantTile, 0, n);
            if constexpr (SrcMask & 4)
                p = readBits(reluTile, 0, n);
            output[r * N + n] = matrixPostprocess<Attr>(acc, accumulatorType, code<dtype_t<D>>, q, p);
        }
    auto maxStep = [&](uint64_t left, uint64_t right) {
        constexpr auto resultType = code<dtype_t<D>>;
        auto x = decodeNumber(resultType, left), y = decodeNumber(resultType, right);
        if constexpr (Attr.MaxAbsEn) {
            x = std::fabs(x);
            y = std::fabs(y);
        }
        if constexpr (Attr.MaxAbsEn) {
            left = encodeNumber(resultType, x, 0, integerCode(resultType));
            right = encodeNumber(resultType, y, 0, integerCode(resultType));
        }
        return minMax(resultType, left, right, true);
    };
    for (std::size_t r = 0; r < rows; ++r) {
        if constexpr (OutMask & 1) {
            auto v = maxStep(output[r * N], output[r * N]);
            for (std::size_t n = 1; n < N; ++n)
                v = maxStep(v, output[r * N + n]);
            if constexpr (SrcMask & 1)
                v = maxStep(readBits(rowIn, r, 0), v);
            rowResults[r] = v;
        }
        if constexpr (OutMask & 2)
            for (int g = 0; g < groups; ++g) {
                auto v = maxStep(output[r * N + g * groupN], output[r * N + g * groupN]);
                for (std::size_t n = g * groupN + 1; n < std::min(N, std::size_t((g + 1) * groupN)); ++n)
                    v = maxStep(v, output[r * N + n]);
                groupResults[r * groups + g] = v;
            }
    }
    if constexpr (OutMask & 1)
        require(static_cast<const void*>(&dst) != static_cast<const void*>(&rowOut), "matrix outputs alias");
    if constexpr (OutMask & 2)
        require(static_cast<const void*>(&dst) != static_cast<const void*>(&groupOut), "matrix outputs alias");
    if constexpr (OutMask == 3)
        require(
            static_cast<const void*>(&rowOut) != static_cast<const void*>(&groupOut), "matrix auxiliary outputs alias");
    produce(dst, N, rows, [&](auto r, auto n) { return output[r * N + n]; });
    if constexpr (OutMask & 1)
        produce(rowOut, 1, rows, [&](auto r, auto) { return rowResults[r]; });
    if constexpr (OutMask & 2)
        produce(groupOut, groups, rows, [&](auto r, auto g) { return groupResults[r * groups + g]; });
}
template <
    FixpAttr Attr, bool Accumulate = false, bool Bias = false, int ScaleMask = 0, typename D, typename A, typename B,
    typename E = NoOperand, typename SA = NoOperand, typename SB = NoOperand>
void matrixBasic(
    D& dst, const A& a, const B& b, std::size_t M, std::size_t N, std::size_t K, const E& extra = noOperand,
    const SA& sa = noOperand, const SB& sb = noOperand)
{
    matrix<Attr, Accumulate, Bias, ScaleMask, 0, 0>(
        dst, a, b, extra, sa, sb, noOperand, noOperand, noOperand, noOperand, noOperand, noOperand, 0, 0, M, N, K);
}

template <typename T>
uint8_t cellByte(const T& tile, std::size_t row, std::size_t index)
{
    if constexpr (bits<dtype_t<T>> == 4)
        return readBits(tile, row, index * 2) | (readBits(tile, row, index * 2 + 1) << 4);
    else
        return readBits(tile, row, index / sizeof(dtype_t<T>)) >> ((index % sizeof(dtype_t<T>)) * 8);
}
template <typename D, typename F>
void produceBytes(D& dst, F fn, unsigned columns = 0)
{
    produce(dst, columns ? columns : dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
        if constexpr (bits<dtype_t<D>> == 4)
            return uint64_t((fn(r, c / 2) >> (c % 2 * 4)) & 15);
        else {
            uint64_t v = 0;
            for (unsigned i = 0; i < sizeof(dtype_t<D>); ++i)
                v |= uint64_t(fn(r, c * sizeof(dtype_t<D>) + i)) << (i * 8);
            return v;
        }
    });
}
template <typename D, typename A, typename B, typename I>
void permute(D& dst, A& a, B& b, I& indices)
{
    constexpr unsigned cellBytes = D::BFractal == BLayout::CubeM16 ? 8 : 4;
    produceBytes(dst, [&](auto r, auto i) {
        auto idx = cellByte(indices, r, i);
        require(idx < 2 * cellBytes, "TPERMUTE byte index is outside both source CELLs");
        return idx < cellBytes ? cellByte(a, r, i / cellBytes * cellBytes + idx) :
                                 cellByte(b, r, i / cellBytes * cellBytes + idx - cellBytes);
    });
}
template <typename D, typename S, typename C>
void shuffle(D& dst, S& src, C& controls, uint64_t control)
{
    const unsigned mode = control & 255, segment = (control >> 8) & 255, boundary = (control >> 16) & 255;
    require((control >> 24) == 0 && mode < 4 && segment < 5 && boundary < 2, "reserved TSHUF control");
    const unsigned width = 2u << segment;
    constexpr unsigned cellRows = S::BFractal == BLayout::CubeM16 ? 16 : 32;
    require(width <= cellRows, "TSHUF segment exceeds CELL rows");
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
        const unsigned shift = readBits(controls, r, c * bits<dtype_t<S>> / 32) & 31;
        const unsigned lane = r % cellRows, local = lane % width, base = r - local;
        int selected = mode == 0 ? int(local) - int(shift) :
                       mode == 1 ? local + shift :
                       mode == 2 ? local ^ shift :
                                   shift % width;
        bool valid = selected >= 0 && selected < int(width) && base + selected < unsigned(src.GetValidRow());
        return valid ? readBits(src, base + selected, c) : boundary == 0 ? readBits(src, r, c) : uint64_t{0};
    });
}
template <typename D, typename A, typename B>
void pack(D& dst, A& a, B& b, uint64_t control)
{
    const unsigned left = control & 255, right = (control >> 8) & 255;
    require(
        control >> 32 == 0 && left >= 1 && left <= 4 && right >= 1 && right <= 4 && left + right <= 4,
        "invalid TPACK byte selection");
    // An assembled destination describes the parent; only this source's raw
    // words belong to the current writer. Do not read/write the next fragment.
    const unsigned columns = ((a.GetValidCol() * sizeof(dtype_t<A>) + 3) / 4) * (4 / sizeof(dtype_t<D>));
    produceBytes(
        dst,
        [&](auto r, auto i) -> uint8_t {
            unsigned pos = i % 4, word = i / 4 * 4;
            return pos < left ? cellByte(a, r, word + pos) : pos < left + right ? cellByte(b, r, word + pos - left) : 0;
        },
        columns);
}
template <typename D, typename S>
void unpack(D& dst, S& src, uint64_t control)
{
    const unsigned offset = control & 255, count = (control >> 8) & 255;
    require(control >> 32 == 0 && count >= 1 && count <= 4 && offset + count <= 4, "invalid TUNPACK byte selection");
    produceBytes(dst, [&](auto r, auto i) -> uint8_t {
        return i % 4 < count ? cellByte(src, r, i / 4 * 4 + offset + i % 4) : 0;
    });
}
template <typename D>
void gprToTile(D& dst, uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3)
{
    const uint64_t words[] = {g0, g1, g2, g3};
    constexpr unsigned rows = D::BFractal == BLayout::CubeM16 ? 16 : 32;
    produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
        uint64_t v = 0;
        if (c >= 32 / rows)
            return v;
        for (unsigned bit = 0; bit < 8; ++bit) {
            unsigned plane = c * 8 + bit;
            v |= ((words[plane / (64 / rows)] >> ((plane % (64 / rows)) * rows + r)) & 1) << bit;
        }
        return v;
    });
}
inline std::mutex memoryAtomicMutex;
inline uint64_t atomicValue(std::string_view op, int type, uint64_t old, uint64_t value, uint64_t expected = 0)
{
    if (op.ends_with("CAS"))
        return old == expected ? value : old;
    if (op.ends_with("EXCH"))
        return value;
    if (op.ends_with("INC"))
        return old >= value ? 0 : old + 1;
    if (op.ends_with("DEC"))
        return old == 0 || old > value ? value : old - 1;
    if (op.ends_with("AND"))
        return old & value;
    if (op.ends_with("XOR"))
        return old ^ value;
    if (op.ends_with("OR"))
        return old | value;
    if (op.ends_with("POPC"))
        return old + 1;
    auto x = decodeNumber(type, old), y = decodeNumber(type, value);
    if (op.ends_with("MAX"))
        return minMax(type, old, value, true);
    if (op.ends_with("MIN"))
        return minMax(type, old, value, false);
    if (integerCode(type))
        return old + value;
    return encodeNumber(type, x + y);
}
template <typename D, typename I, typename V, typename E = NoOperand>
void memoryAtomic(
    std::string_view op, D& dst, uint64_t base, const I& indices, const V& operand, unsigned columns, unsigned rows,
    IndexedByteRange range, const E& expected = noOperand)
{
    using Value = dtype_t<V>;
    columns = I::ValidCol > 0 ? I::ValidCol : columns;
    rows = I::ValidRow > 0 ? I::ValidRow : rows;
    require(columns <= I::Cols && rows <= I::Rows, "atomic operation exceeds index shape");
    struct Update {
        uintptr_t address;
        uint64_t value, expected;
    };
    std::vector<Update> updates;
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < columns; ++c) {
            auto address = indexedAddress(
                reinterpret_cast<const void*>(base), int64_t(number(read(indices, r, c))), range, sizeof(Value));
            require(address % sizeof(Value) == 0, "atomic address is misaligned");
            uint64_t exp = 0;
            if constexpr (!std::is_same_v<E, NoOperand>)
                exp = readBits(expected, r, c);
            updates.push_back({address, readBits(operand, r, c), exp});
        }
    std::vector<uint64_t> observed(updates.size());
    {
        std::lock_guard lock(memoryAtomicMutex);
        for (unsigned i = 0; i < updates.size(); ++i) {
            auto u = updates[i];
            uint64_t old = 0;
            std::memcpy(&old, reinterpret_cast<const void*>(u.address), sizeof(Value));
            observed[i] = old;
            auto next = atomicValue(op, code<Value>, old, u.value, u.expected);
            std::memcpy(reinterpret_cast<void*>(u.address), &next, sizeof(Value));
        }
    }
    if constexpr (!std::is_same_v<D, const NoOperand> && !std::is_same_v<D, NoOperand>)
        produce(dst, columns, rows, [&](auto r, auto c) { return observed[r * columns + c]; });
}
template <typename I>
void memoryPopcount(uint64_t base, const I& indices, unsigned columns, unsigned rows, IndexedByteRange range)
{
    columns = I::ValidCol > 0 ? I::ValidCol : columns;
    rows = I::ValidRow > 0 ? I::ValidRow : rows;
    std::vector<uintptr_t> addresses;
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < columns; ++c) {
            auto p =
                indexedAddress(reinterpret_cast<const void*>(base), int64_t(number(read(indices, r, c))), range, 4);
            require(p % 4 == 0, "MSCATTER.POPC address is misaligned");
            addresses.push_back(p);
        }
    std::lock_guard lock(memoryAtomicMutex);
    for (auto p : addresses) {
        uint32_t value;
        std::memcpy(&value, reinterpret_cast<const void*>(p), 4);
        ++value;
        std::memcpy(reinterpret_cast<void*>(p), &value, 4);
    }
}
template <int Mode, int Mask, bool Invert = false, typename G, typename S, typename M = NoOperand>
void atomicStore(G& dst, const S& src, const M& mask = noOperand)
{
    if (!(Mask & (8u >> threadIndex())))
        return;
    constexpr std::string_view names[] = {"ADD", "MIN", "MAX", "AND", "OR", "XOR", "INC", "DEC"};
    using E = dtype_t<S>;
    std::vector<std::pair<void*, uint64_t>> writes;
    for (int r = 0; r < src.GetValidRow(); ++r)
        for (int c = 0; c < src.GetValidCol(); ++c) {
            if constexpr (!std::is_same_v<M, NoOperand>) {
                const auto* p = reinterpret_cast<const std::byte*>(&mask);
                if ((std::to_integer<unsigned>(p[predicateIndex<S>(r, c)]) != 0) == Invert)
                    continue;
            }
            auto* address = reinterpret_cast<std::byte*>(dst.data()) + r * dst.GetStrideBytes(3) + c * sizeof(E);
            require(reinterpret_cast<uintptr_t>(address) % sizeof(E) == 0, "atomic TSTORE address is misaligned");
            writes.emplace_back(address, readBits(src, r, c));
        }
    std::lock_guard lock(memoryAtomicMutex);
    for (auto [address, value] : writes) {
        uint64_t old = 0;
        std::memcpy(&old, address, sizeof(E));
        auto next = atomicValue(names[Mode], code<E>, old, value);
        std::memcpy(address, &next, sizeof(E));
    }
}
template <typename G, typename S>
void storeCount(G& dst, const S& src, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        const auto r = i / src.GetValidCol(), c = i % src.GetValidCol();
        auto v = readBits(src, r, c);
        std::memcpy(
            reinterpret_cast<std::byte*>(dst.data()) + r * dst.GetStrideBytes(3) + c * sizeof(dtype_t<S>), &v,
            sizeof(dtype_t<S>));
    }
}
// A metadata-only adapter: constructing a view must not allocate a second Tile.
template <typename Descriptor, typename Carrier>
struct CarrierTile {
    using DType = typename Descriptor::DType;
    static constexpr auto Loc = Descriptor::Loc;
    static constexpr auto BFractal = Descriptor::BFractal;
    static constexpr auto SFractal = Descriptor::SFractal;
    static constexpr bool IsCubeLayout = Descriptor::IsCubeLayout;
    static constexpr int Rows = Descriptor::Rows, Cols = Descriptor::Cols;
    static constexpr int InnerRows = Descriptor::InnerRows, InnerCols = Descriptor::InnerCols;
    static constexpr int LogicalTileBytes = Descriptor::LogicalTileBytes;
    static constexpr int ValidRow = Descriptor::ValidRow, ValidCol = Descriptor::ValidCol;
    Carrier& storage;
    std::size_t offset;
    CarrierTile(Carrier& carrier, std::size_t base = 0) : storage(carrier), offset(base) {}
    int GetValidRow() const { return ValidRow; }
    int GetValidCol() const { return ValidCol; }
    auto& data() { return storage; }
    const auto& data() const { return storage; }
    auto GetRangeBase() const { return offset; }
};
template <typename D, typename G>
void loadLocalAssembly(D& dst, const G& src)
{
    using W = typename D::Writer;
    auto& parent = detail::LocalCubeAssemblyAccess::parent_data(dst);
    CarrierTile<W, std::remove_reference_t<decltype(parent)>> view(parent, D::OffsetCells);
    load(view, src);
}
template <typename P, int C, int VR, int VC, typename G>
void store(G& dst, const LocalCubeAssemblyResult<P, C, VR, VC>& src)
{
    using Desc = Tile<P::Loc, typename P::DType, 32, C, BLayout::CubeM32, VR, VC>;
    CarrierTile<Desc, const std::remove_reference_t<decltype(src.data())>> view(src.data());
    store(dst, view);
}

template <typename E>
uint64_t globalElement(const void* base, std::size_t index)
{
    uint64_t raw = 0;
    auto p = static_cast<const std::byte*>(base);
    if constexpr (bits<E> == 4)
        return (std::to_integer<unsigned>(p[index / 2]) >> (index % 2 * 4)) & 15;
    else {
        std::memcpy(&raw, p + index * sizeof(E), sizeof(E));
        return raw;
    }
}
template <WeightLayoutEnum Layout, int Mask = 1, typename D, typename G>
void weightLoad(D& dst, const G& src, WeightTLOADParams params)
{
    if (skipSharedProducer(dst, Mask))
        return;
    const unsigned ci = params.shape_word & 65535, co = (params.shape_word >> 16) & 65535;
    const unsigned kh = (params.shape_word >> 32) & 255, kw = (params.shape_word >> 40) & 255;
    const uint64_t nstart = uint32_t(params.start_word), kstart = params.start_word >> 32;
    constexpr unsigned c0 = 256 / bits<dtype_t<G>>;
    const unsigned channels = (ci + c0 - 1) / c0 * c0;
    require((params.shape_word >> 48) == 0 && ci && co && kh && kw, "invalid weight TLOAD shape word");
    require(
        nstart + dst.GetValidRow() <= co && kstart + dst.GetValidCol() <= uint64_t(kh) * kw * channels,
        "weight TLOAD window exceeds source shape");
    require(kstart % c0 == 0 && dst.GetValidCol() % c0 == 0, "weight TLOAD K window is not C0 aligned");
    produce(
        dst, dst.GetValidCol(), dst.GetValidRow(),
        [&](auto r, auto c) {
            auto k = kstart + c, n = nstart + r;
            auto channel = k % channels, spatial = k / channels;
            if (channel >= ci)
                return uint64_t{0};
            auto index =
                Layout == OHWI2NK ? (n * kh * kw + spatial) * ci + channel : (n * ci + channel) * kh * kw + spatial;
            return globalElement<dtype_t<G>>(src.data(), index);
        },
        Mask);
}
inline thread_local std::array<unsigned long, 64> sharedSlots{};
template <unsigned Slot, int Mask, typename T>
void publishSlot(SharedTile<T>& dst)
{
    sharedSlots[Slot] = dst.handle();
    if (!currentCore || !(Mask & (8u >> threadIndex())))
        return;
    auto& storage = sharedStorage.at(dst.handle());
    resolveShared(storage);
    require(storage.published, "Shared slot producer did not publish its parent");
    std::lock_guard lock(currentCore->mutex);
    currentCore->sharedSlots[Slot] = storage.payload;
    currentCore->changed.notify_all();
}
template <unsigned Slot, typename T>
void acquireSlot(SharedTile<T>& dst)
{
    if (!currentCore) {
        dst.handle_ref() = sharedSlots[Slot];
        return;
    }
    auto& core = *currentCore;
    std::unique_lock lock(core.mutex);
    core.slotWaiters |= 1u << threadIndex();
    core.changed.notify_all();
    core.changed.wait(lock, [&] {
        return core.sharedSlots.contains(Slot) || core.error || (core.finished | core.slotWaiters) == 15;
    });
    core.slotWaiters &= ~(1u << threadIndex());
    if (core.error)
        std::rethrow_exception(core.error);
    require(core.sharedSlots.contains(Slot), "Shared slot has no producer");
    const auto& payload = core.sharedSlots.at(Slot);
    require(payload.size() == capacity<SharedTile<T>>(), "Shared slot capacity differs from its consumer");
    dst.handle_ref() = nextSharedHandle++;
    auto& storage = sharedStorage[dst.handle()];
    storage.payload = payload;
    storage.published = true;
}
template <WeightLayoutEnum Layout, unsigned Slot, int Mask, typename T, typename G>
void weightLoadSlot(const G& src, WeightTLOADParams params)
{
    SharedTile<T> dst;
    weightLoad<Layout, Mask>(dst, src, params);
    publishSlot<Slot, Mask>(dst);
}
template <unsigned SlotA, typename A, unsigned SlotB, typename B, FixpAttr Attr, typename D>
void matrixSlot(D& dst)
{
    SharedTile<typename A::LocalTileType> a;
    SharedTile<typename B::LocalTileType> b;
    acquireSlot<SlotA>(a);
    acquireSlot<SlotB>(b);
    matrixBasic<Attr>(dst, a, b, A::ValidRow, Attr.TransB ? B::ValidCol : B::ValidRow, A::ValidCol);
}
template <LayoutCvtEnum Order = NORM, typename D, typename G>
void imageToColumn(D& dst, const G& src, TIMG2COLParams params, std::size_t groupRows = 0, unsigned mask = 15)
{
    if (skipSharedProducer(dst, mask))
        return;
    const auto p = decode_timg2col_params(params);
    require(is_timg2col_params_base_legal(params), "invalid TIMG2COL parameter words");
    const int64_t hout =
        (int64_t(p.input_h) + p.pad_top + p.pad_bottom - (p.kernel_h - 1) * p.dilation_h - 1) / p.conv_stride_h + 1;
    const int64_t wout =
        (int64_t(p.input_w) + p.pad_left + p.pad_right - (p.kernel_w - 1) * p.dilation_w - 1) / p.conv_stride_w + 1;
    require(hout > 0 && wout > 0, "TIMG2COL has no output spatial positions");
    constexpr unsigned c0 = 256 / bits<dtype_t<G>>;
    const uint64_t channels = (p.cin + c0 - 1) / c0 * c0;
    const unsigned shardRows = D::BFractal == BLayout::CubeM16 ? 16 : 32;
    std::size_t offset = 0, rows = dst.GetValidRow();
    const auto total = groupRows ? groupRows : rows;
    if constexpr (D::Loc != Location::Shared) {
        offset = threadIndex() * shardRows;
        rows = offset < total ? std::min<std::size_t>(shardRows, total - offset) : 0;
    }
    require(
        p.row_start + total <= uint64_t(hout) * wout &&
            p.col_start + dst.GetValidCol() <= p.kernel_h * p.kernel_w * channels,
        "TIMG2COL requested rectangle exceeds output shape");
    require(p.col_start % c0 == 0 && dst.GetValidCol() % c0 == 0, "TIMG2COL K window is not C0 aligned");
    produce(
        dst, dst.GetValidCol(), rows,
        [&](auto r, auto c) {
            uint64_t expandedRow = p.row_start + offset + r, expandedCol = p.col_start + c;
            auto kernel = expandedCol / channels, channel = expandedCol % channels;
            int64_t h = (expandedRow / wout) * p.conv_stride_h + (kernel / p.kernel_w) * p.dilation_h - p.pad_top;
            int64_t w = (expandedRow % wout) * p.conv_stride_w + (kernel % p.kernel_w) * p.dilation_w - p.pad_left;
            if (h < 0 || w < 0 || h >= p.input_h || w >= p.input_w || channel >= p.cin)
                return uint64_t{0};
            uint64_t idx =
                Order == DN2ND ? (channel * p.input_h + h) * p.input_w + w : (h * p.input_w + w) * p.cin + channel;
            return globalElement<dtype_t<G>>(src.data(), idx);
        },
        mask);
}
template <LayoutCvtEnum Order, unsigned Slot, int Mask, typename T, typename G>
void imageToColumnSlot(const G& src, TIMG2COLParams params)
{
    SharedTile<T> dst;
    imageToColumn<Order>(dst, src, params, 0, Mask);
    publishSlot<Slot, Mask>(dst);
}
template <bool Columns, typename G, typename... D>
void loadMultiple(const G& src, D&... destinations)
{
    std::size_t offset = 0;
    auto one = [&](auto& dst) {
        produce(dst, dst.GetValidCol(), dst.GetValidRow(), [&](auto r, auto c) {
            auto rr = r + (Columns ? 0 : offset), cc = c + (Columns ? offset : 0);
            uint64_t v = 0;
            std::memcpy(
                &v,
                reinterpret_cast<const std::byte*>(src.data()) + rr * src.GetStrideBytes(3) +
                    cc * src.GetStrideBytes(4),
                sizeof(dtype_t<G>));
            return convertRaw(code<dtype_t<plain_t<decltype(dst)>>>, code<dtype_t<G>>, v, 0, false);
        });
        offset += Columns ? dst.GetValidCol() : dst.GetValidRow();
    };
    (one(destinations), ...);
}
template <typename G, typename... S>
void storeMultiple(G& dst, const S&... sources)
{
    std::size_t offset = 0;
    auto one = [&](const auto& src) {
        for (unsigned r = 0; r < unsigned(src.GetValidRow()); ++r)
            for (unsigned c = 0; c < unsigned(src.GetValidCol()); ++c) {
                auto v = readBits(src, r, c);
                std::memcpy(
                    reinterpret_cast<std::byte*>(dst.data()) + (r + offset) * dst.GetStrideBytes(3) +
                        c * dst.GetStrideBytes(4),
                    &v, sizeof(dtype_t<G>));
            }
        offset += src.GetValidRow();
    };
    (one(sources), ...);
}

template <int Mask, typename D, typename S>
void groupMove(D& dst, uint64_t peer, const S& src)
{
    require(peer < 4, "GMOV peer identity is outside Core4");
    if (!currentCore) {
        require(peer == threadIndex(), "GMOV peer exchange requires linx_cpu::runCore");
        if (Mask & (8u >> threadIndex()))
            copy(dst, src);
        return;
    }
    auto& core = *currentCore;
    std::unique_lock lock(core.mutex);
    if (core.error)
        std::rethrow_exception(core.error);
    require(core.finished == 0, "GMOV participant exited before the collective");
    const auto generation = core.generation;
    auto& entry = core.copies[threadIndex()];
    entry.destination = bytes(dst);
    entry.peer = peer;
    entry.mask = Mask;
    entry.source.assign(bytes(src), bytes(src) + capacity<S>());
    if (++core.arrived == 4) {
        for (unsigned pe = 0; pe < 4; ++pe) {
            const auto& item = core.copies[pe];
            const auto& from = core.copies[item.peer];
            require(from.source.size() == item.source.size(), "GMOV peers disagree on carrier size");
            if (item.mask & (8u >> pe))
                std::memcpy(item.destination, from.source.data(), item.source.size());
        }
        core.arrived = 0;
        ++core.generation;
        core.changed.notify_all();
    } else {
        core.changed.wait(lock, [&] { return core.generation != generation || core.error; });
        if (core.error)
            std::rethrow_exception(core.error);
    }
}

template <int ParentSize, typename Sub>
auto outputView(region::TileArrayOutputRef<Sub>& dst)
{
    auto& carrier = dst.template parent_data<ParentSize>();
    return CarrierTile<Sub, std::remove_reference_t<decltype(carrier)>>(carrier, dst.range_base_units());
}
template <int Op, typename D, typename S>
void regionUnary(D& dst, S& src)
{
    if constexpr (Op >= 64)
        reduce<Op>(dst, src);
    else
        unary<Op>(dst, src);
}
template <bool First, int Op, typename D, typename V, typename T>
void mixedBinary(D& dst, V& view, T& tile)
{
    if constexpr (First)
        binary<Op>(dst, view, tile);
    else
        binary<Op>(dst, tile, view);
}
template <typename Basis, typename Carrier>
auto predicateView(Carrier& carrier)
{
    constexpr int rows = Basis::BFractal == BLayout::CubeM16 ? 16 : 32;
    constexpr int columns = Basis::Cols < 128 / rows ? 128 / rows : Basis::Cols;
    using Descriptor = Tile<Location::Vec, uint8_t, rows, columns, Basis::BFractal, Basis::ValidRow, Basis::ValidCol>;
    return CarrierTile<Descriptor, Carrier>(carrier);
}
template <bool Atomic, bool Invert, typename G, typename I, typename V, typename M>
void predicateScatter(G& dst, const I& indices, const V& values, const M& active, IndexedByteRange range)
{
    std::vector<std::pair<uintptr_t, uint64_t>> writes;
    for (int r = 0; r < values.GetValidRow(); ++r)
        for (int c = 0; c < values.GetValidCol(); ++c) {
            if (maskBit<V>(active, r, c) == Invert)
                continue;
            auto address = indexedAddress(dst.data(), int64_t(number(read(indices, r, c))), range, sizeof(dtype_t<V>));
            if constexpr (Atomic)
                require(address % sizeof(dtype_t<V>) == 0, "masked atomic address is misaligned");
            writes.emplace_back(address, readBits(values, r, c));
        }
    std::lock_guard lock(memoryAtomicMutex);
    for (auto [address, v] : writes) {
        if constexpr (Atomic) {
            uint64_t old = 0;
            std::memcpy(&old, reinterpret_cast<const void*>(address), sizeof(dtype_t<V>));
            v = atomicValue("ADD", code<dtype_t<V>>, old, v);
        }
        std::memcpy(reinterpret_cast<void*>(address), &v, sizeof(dtype_t<V>));
    }
}

template <typename D, typename S>
void sharedNDtoCube(D& dst, const S& src, unsigned rows, unsigned columns)
{
    const auto* base = [&] {
        if constexpr (requires { src.parent(); })
            return bytes(src.parent()) + (src.GetRangeBase() + S::Offset) * 128;
        else
            return bytes(src);
    }();
    constexpr auto stride = [] {
        if constexpr (requires { typename S::ParentTile; })
            return S::ParentTile::Cols;
        else
            return S::Cols;
    }();
    produce(dst, columns, rows, [&](auto r, auto c) { return globalElement<dtype_t<D>>(base, r * stride + c); });
}

template <typename D>
void publishShared(D& dst, unsigned mask, bool publish)
{
    auto& storage = sharedStorage.at(dst.handle());
    if (currentCore) {
        auto& core = *currentCore;
        std::unique_lock lock(core.mutex);
        if (core.error)
            std::rethrow_exception(core.error);
        const auto pe = threadIndex();
        const auto ordinal = core.sharedSequence[mask][pe]++;
        auto& exchange = core.sharedExchanges[{mask, ordinal}];
        unsigned participants = 0;
        for (unsigned worker = 0; worker < 4; ++worker)
            if (mask & (8u >> worker))
                participants |= 1u << worker;
        if (participants & (1u << pe)) {
            exchange.sources[pe] = storage.payload;
            exchange.arrived |= 1u << pe;
        }
        if (!exchange.ready && exchange.arrived == participants) {
            const bool singleton = std::popcount(mask) == 1;
            unsigned issuer = std::countr_zero(participants);
            auto result = exchange.sources[issuer];
            if constexpr (is_assemble_v<D>) {
                if (!singleton)
                    issuer = viewOffset(dst) / (128u << (D::WriterSizeCode - 1)) % 4;
                if (participants & (1u << issuer))
                    result = exchange.sources[issuer];
            } else if (singleton)
                result = exchange.sources[issuer];
            else
                for (unsigned worker = 0; worker < 4; ++worker)
                    if (participants & (1u << worker)) {
                        require(
                            exchange.sources[worker].size() == result.size(), "Shared producers disagree on capacity");
                        auto first = result.size() * worker / 4, last = result.size() * (worker + 1) / 4;
                        std::copy(
                            exchange.sources[worker].begin() + first, exchange.sources[worker].begin() + last,
                            result.begin() + first);
                    }
            exchange.result = std::move(result);
            exchange.ready = true;
            core.changed.notify_all();
        }
        if (participants & (1u << pe)) {
            core.changed.wait(lock, [&] {
                return exchange.ready || core.error || (core.finished & participants & ~exchange.arrived);
            });
            if (core.error)
                std::rethrow_exception(core.error);
            require(exchange.ready, "Shared producer participant exited before publication");
            require(exchange.result.size() == storage.payload.size(), "Shared participants disagree on capacity");
            storage.payload = exchange.result;
        } else {
            // A masked-out producer is a no-op and cannot wait for selected
            // PEs. Resolve the generation lazily if this PE later consumes it.
            storage.context = &core;
            storage.pending = &exchange;
            storage.participants = participants;
        }
    }
    if constexpr (is_assemble_v<D>)
        storage.published = D::LAST && publish;
    else
        storage.published = publish;
}
} // namespace pto::linx_cpu
#endif
