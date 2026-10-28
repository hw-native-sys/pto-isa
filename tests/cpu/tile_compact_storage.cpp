/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
constexpr uint8_t CANARY = 0xa5;
constexpr size_t UB_BYTES = 256 * 1024;
size_t checks = 0;

void require(bool condition, const char* message)
{
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}

template <
    typename T, int Rows, int ValidRows = DYNAMIC, CompactMode Mode = CompactMode::RowPlusOne,
    auto Pad = PadValue::Null, int ValidCols = DYNAMIC>
using NZ = Tile<
    TileType::Vec, T, Rows, 128 / sizeof(T), BLayout::ColMajor, ValidRows, ValidCols, SLayout::RowMajor,
    TileConfig::fractalABSize, Pad, Mode>;

// Independent oracle for the existing CPU representation: Rows is the physical
// panel stride. This storage-only regression does not assert native compact pitch.
template <typename Data>
size_t legacyOffset(unsigned r, unsigned c)
{
    constexpr size_t PANEL = 32 / sizeof(typename Data::DType);
    return (c / PANEL) * Data::Rows * PANEL + r * PANEL + c % PANEL;
}

int value(unsigned r, unsigned c) { return static_cast<int>((r * 13 + c * 7) % 41) - 20; }

template <typename Data>
struct Guarded {
    using T = typename Data::DType;
    static constexpr size_t GUARD = 32 / sizeof(T);
    Data tile;
    std::shared_ptr<std::vector<T>> backing = std::make_shared<std::vector<T>>(Data::GetSizeInUnits() + 2 * GUARD);

    Guarded(unsigned rows, unsigned cols)
    {
        if constexpr (Data::ValidRow == DYNAMIC)
            tile.SetValidRow(rows);
        if constexpr (Data::ValidCol == DYNAMIC)
            tile.SetValidCol(cols);
        require(rows == tile.GetValidRow() && cols == tile.GetValidCol(), "fixture valid shape mismatch");
        NPUMemoryModel::Instance().RegisterHostStorage(backing);
        TASSIGN(tile, reinterpret_cast<uintptr_t>(backing->data() + GUARD));
        std::memset(backing->data(), CANARY, backing->size() * sizeof(T));
    }

    template <typename Value>
    void initialize(Value fn)
    {
        std::memset(backing->data(), CANARY, backing->size() * sizeof(T));
        for (unsigned r = 0; r < tile.GetValidRow(); ++r)
            for (unsigned c = 0; c < tile.GetValidCol(); ++c)
                tile.data()[legacyOffset<Data>(r, c)] = static_cast<T>(fn(r, c));
    }

    template <typename Value>
    void verify(Value fn, const char* message)
    {
        std::vector<T> expected(backing->size());
        std::memset(expected.data(), CANARY, expected.size() * sizeof(T));
        for (unsigned r = 0; r < tile.GetValidRow(); ++r)
            for (unsigned c = 0; c < tile.GetValidCol(); ++c)
                expected[GUARD + legacyOffset<Data>(r, c)] = static_cast<T>(fn(r, c));
        require(std::memcmp(backing->data(), expected.data(), expected.size() * sizeof(T)) == 0, message);
    }
};

template <typename T>
void checkCapacity()
{
    using Dynamic = NZ<T, 17>;
    using Legacy = NZ<T, 33, 32>;
    using Dynamic33 = NZ<T, 33>;
    using Static33 = NZ<T, 33, 33>;
    using Static64 = NZ<T, 64, 19>;
    using Plain = NZ<T, 17, DYNAMIC, CompactMode::Null>;
    using Normal = NZ<T, 17, DYNAMIC, CompactMode::Normal>;
    static_assert(Dynamic::Numel == 17 * Dynamic::Cols && Dynamic::StorageNumel == 33 * Dynamic::Cols);
    static_assert(Legacy::StorageNumel == 33 * Legacy::Cols);
    static_assert(Dynamic33::StorageNumel == 49 * Dynamic33::Cols);
    static_assert(Static33::StorageNumel == 49 * Static33::Cols);
    static_assert(Static64::StorageNumel == 65 * Static64::Cols);
    static_assert(Plain::StorageNumel == Plain::Numel && Normal::StorageNumel == Normal::Numel);
    static_assert(Dynamic::StorageBytes == Dynamic::StorageNumel / Dynamic::kPackedElementsPerByte * sizeof(T));
    static_assert(Dynamic::GetSizeInBytes() == Dynamic::StorageBytes);
    for (bool constFirst : {false, true}) {
        Dynamic tile;
        tile.SetValidShape(17, Dynamic::Cols);
#ifndef __PTO_AUTO__
        TASSIGN(tile, 32);
#endif
        const Dynamic& view = tile;
        auto* bytes = reinterpret_cast<uint8_t*>(constFirst ? view.data() : tile.data());
        bytes[Dynamic::StorageBytes - 1] = 0x3d;
        require(
            reinterpret_cast<const uint8_t*>(view.data())[Dynamic::StorageBytes - 1] == 0x3d,
            "last physical byte missing");
    }
    auto& memory = NPUMemoryModel::Instance();
    auto* last = memory.GetPointer<Dynamic>(UB_BYTES - Dynamic::StorageBytes);
    require(
        reinterpret_cast<char*>(last) + Dynamic::StorageBytes == memory.GetUBBase() + UB_BYTES,
        "exact padded UB capacity rejected");
    last = memory.GetPointer<Legacy>(UB_BYTES - Legacy::StorageBytes);
    require(
        reinterpret_cast<char*>(last) + Legacy::StorageBytes == memory.GetUBBase() + UB_BYTES,
        "exact legacy UB capacity rejected");
    auto backing = std::make_shared<std::vector<T>>(Dynamic::GetSizeInUnits());
    memory.RegisterHostStorage(backing);
    require(
        memory.ResolveAssignedAddress<Dynamic>(reinterpret_cast<uintptr_t>(backing->data())) == backing->data(),
        "exact host capacity rejected");
}

template <typename Data>
void checkLegacyOperations(unsigned rows)
{
    Guarded<Data> a(rows, Data::Cols), b(rows, Data::Cols), out(rows, Data::Cols);
    a.initialize(value);
    b.initialize([](unsigned r, unsigned c) { return value(r, c) + 3; });
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < Data::Cols; ++c) {
            require(
                GetTileElementOffset<Data>(r, c) == legacyOffset<Data>(r, c),
                "descriptor changed the legacy Rows stride");
            require(a.tile.GetElement(r, c) == value(r, c), "element accessor changed raw legacy layout");
        }
    TADD(out.tile, a.tile, b.tile);
    out.verify([](unsigned r, unsigned c) { return 2 * value(r, c) + 3; }, "legacy TADD or guards changed");
    out.initialize([](unsigned, unsigned) { return 0; });
    TMIN(out.tile, a.tile, b.tile);
    out.verify(value, "legacy shared binary helper or guards changed");
    TEXPANDS(out.tile, static_cast<typename Data::DType>(7));
    out.verify([](unsigned, unsigned) { return 7; }, "legacy scalar helper or guards changed");

    std::ostringstream text;
    auto* previous = std::cout.rdbuf(text.rdbuf());
    TPRINT(a.tile);
    std::cout.rdbuf(previous);
    std::istringstream parsed(text.str());
    std::string name, shape;
    parsed >> name >> shape;
    require(
        name == "TPRINT" && shape == std::to_string(rows) + "x" + std::to_string(Data::Cols),
        "legacy TPRINT shape changed");
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < Data::Cols; ++c) {
            int actual = 0;
            require(static_cast<bool>(parsed >> actual) && actual == value(r, c), "legacy TPRINT values changed");
        }
}

template <int Rows, int ValidRows>
void checkLegacyFill(unsigned rows)
{
    using Data = NZ<int32_t, Rows, ValidRows, CompactMode::RowPlusOne, PadValue::Zero>;
    Guarded<Data> guarded(rows, Data::Cols);
    guarded.initialize(value);
    auto expected = *guarded.backing;
    for (unsigned r = rows; r < Rows; ++r)
        for (unsigned c = 0; c < Data::Cols; ++c)
            expected[Guarded<Data>::GUARD + legacyOffset<Data>(r, c)] = 0;
    TFILLPAD(guarded.tile, guarded.tile);
    require(
        std::memcmp(guarded.backing->data(), expected.data(), expected.size() * sizeof(int32_t)) == 0,
        "legacy whole-Rows TFILLPAD or guards changed");
}

template <typename Data>
void checkFifoBacking()
{
    Data tile;
    if constexpr (Data::ValidRow == DYNAMIC)
        tile.SetValidRow(17);
    if constexpr (Data::ValidCol == DYNAMIC)
        tile.SetValidCol(Data::Cols);
    cpu_pipe::EnsureTileStorage(tile);
#ifdef __PTO_AUTO__
    // Tile::data() has already allocated its physical envelope in AUTO mode.
    constexpr size_t UNITS = Data::GetSizeInUnits();
    static_assert(!std::is_same_v<Data, NZ<float4_e2m1x2_t, 17>> || UNITS == 2112); // DType objects, not FP4 elements.
#else
    constexpr size_t OLD_UNITS = static_cast<size_t>(Data::Rows) * Data::Cols;
    constexpr size_t UNITS = Data::GetSizeInUnits() > OLD_UNITS ? Data::GetSizeInUnits() : OLD_UNITS;
    static_assert(
        !std::is_same_v<Data, NZ<float4_e2m1x2_t, 17>> || UNITS == 2176); // Manual backing keeps legacy units.
#endif
    auto* bytes = reinterpret_cast<uint8_t*>(tile.data());
    NPUMemoryModel::Instance().CheckAccess(bytes, UNITS * sizeof(typename Data::DType) - 1, 1);
    bytes[UNITS * sizeof(typename Data::DType) - 1] = 0x5c;
    require(bytes[UNITS * sizeof(typename Data::DType) - 1] == 0x5c, "FIFO backing shrank");
}

template <typename Data>
void rejectCapacity(bool host, size_t bytes)
{
    auto& memory = NPUMemoryModel::Instance();
    if (host) {
        auto backing = std::make_shared<std::vector<uint8_t>>(bytes);
        memory.RegisterHostStorage(backing);
        memory.ResolveAssignedAddress<Data>(reinterpret_cast<uintptr_t>(backing->data()));
    } else {
        memory.GetPointer<Data>(UB_BYTES - bytes);
    }
}
} // namespace

int main(int argc, char** argv)
{
#if defined(_WIN32)
    require(_putenv_s("PTO_CPU_SIM_UB_BYTES", "262144") == 0, "failed to fix UB capacity");
#else
    require(setenv("PTO_CPU_SIM_UB_BYTES", "262144", 1) == 0, "failed to fix UB capacity");
#endif
    NPU_MEMORY_INIT(NPUArch::A5);
    if (argc > 1) {
        const std::string mode = argv[1];
        const bool host = mode.find("host") != std::string::npos;
        if (mode.find("legacy") != std::string::npos) {
            using Data = NZ<int32_t, 33, 32>;
            rejectCapacity<Data>(host, Data::StorageBytes - 32);
        } else if (mode.find("dynamic") != std::string::npos) {
            using Data = NZ<int32_t, 33>;
            rejectCapacity<Data>(host, 33 * 128);
        } else {
            using Data = NZ<int32_t, 32>;
            rejectCapacity<Data>(host, 32 * 128);
        }
        return 3;
    }
    checkCapacity<int8_t>();
    checkCapacity<int16_t>();
    checkCapacity<int32_t>();
    checkCapacity<float>();
    checkCapacity<float4_e2m1x2_t>();
    checkLegacyOperations<NZ<int32_t, 33>>(32);
    checkLegacyOperations<NZ<int32_t, 33>>(19);
    checkLegacyOperations<NZ<int32_t, 64>>(19);
    checkLegacyOperations<NZ<int32_t, 64, 19>>(19);
    checkLegacyOperations<NZ<int32_t, 64, DYNAMIC, CompactMode::Null>>(19);
    checkLegacyOperations<NZ<int32_t, 64, DYNAMIC, CompactMode::Normal>>(19);
    checkLegacyFill<33, 32>(32);
    checkLegacyFill<64, DYNAMIC>(19);
    checkFifoBacking<NZ<int8_t, 17>>();
    checkFifoBacking<NZ<float4_e2m1x2_t, 17>>();
    std::cout << "RowPlusOne storage and legacy CPU checks passed: " << checks << '\n';
}
