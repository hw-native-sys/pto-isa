/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#if defined(__CPU_SIM)
#include <array>
#include <chrono>
#include <limits>
#include <stdexcept>
using namespace pto;
static void check(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <typename Function>
static void rejects(Function fn)
{
    bool caught = false;
    try {
        fn();
    } catch (const std::exception&) {
        caught = true;
    }
    check(caught, "invalid runtime operation was accepted");
}

static void launchAndMemory()
{
    std::array<std::byte, 512> storage{};
    std::array<linx_cpu::MemoryRegion, 1> regions{{{0, storage}}};
    for (unsigned count = 1; count <= 4; ++count) {
        std::array<unsigned, 4> counters{};
        linx_cpu::runCore(
            [&] {
                auto pe = get_thread_idx();
                for (unsigned generation = 1; generation <= 16; ++generation) {
                    counters[pe] = generation;
                    linx_cpu::syncCore();
                    for (unsigned i = 0; i < count; ++i)
                        check(counters[i] == generation, "barrier did not publish peer writes");
                    linx_cpu::syncCore();
                }
                using V = VecTileM32<uint32_t, 32, 1>;
                V source, result;
                TEXPANDS(source, 10u + pe);
                global_tensor<uint32_t, RowMajor<32, 1>> gm(reinterpret_cast<uint32_t*>(pe * 128));
                TSTORE(gm, source);
                linx_cpu::syncCore();
                global_tensor<uint32_t, RowMajor<32, 1>> peer(reinterpret_cast<uint32_t*>(((pe + 1) % count) * 128));
                TLOAD(result, peer);
                for (unsigned row = 0; row < 32; ++row)
                    check(linx_cpu::readBits(result, row, 0) == 10 + (pe + 1) % count, "mapped GM peer read");
                rejects([&] { linx_cpu::globalAddress(nullptr, 508, 8); });
                rejects([&] { linx_cpu::globalAddress(nullptr, 520, 4); });
                if (count == 1 || count == 4) {
                    GMOV<15>(result, (pe + 1) % count, source);
                    check(linx_cpu::readBits(result, 0, 0) == 10 + (pe + 1) % count, "GMOV active peer");
                }
            },
            {count, regions});
    }
    rejects([] { linx_cpu::runCore([] {}, {0, {}}); });
    rejects([] { linx_cpu::globalAddress(nullptr, 0, 4); });
    std::array<linx_cpu::MemoryRegion, 2> adjacent{
        {{0, std::span(storage).first(256)}, {256, std::span(storage).subspan(256)}}};
    linx_cpu::runCore(
        [&] {
            check(
                linx_cpu::globalAddress(reinterpret_cast<void*>(256), 0, 4) == storage.data() + 256,
                "adjacent GM mapping boundary");
        },
        {1, adjacent});
    adjacent[1].address = 128;
    rejects([&] { linx_cpu::runCore([] {}, {1, adjacent}); });
    rejects([] { linx_cpu::globalAddress(reinterpret_cast<void*>(UINTPTR_MAX - 1), 4, 1); });
    rejects([] {
        linx_cpu::runCore([] {
            if (get_thread_idx() == 2)
                throw std::runtime_error("worker failure");
            linx_cpu::syncCore();
        });
    });
    // A divergent barrier must wake and fail even if a peer exits normally.
    rejects([] {
        linx_cpu::runCore([] {
            if (get_thread_idx() != 0)
                linx_cpu::syncCore();
        });
    });
}

static void slotGenerations()
{
    using Plane = Tile<Location::Vec, float, 4, 8>;
    linx_cpu::runCore([] {
        const auto pe = get_thread_idx();
        if (pe == 0) {
            for (unsigned generation = 0; generation < 8; ++generation) {
                float input[32];
                for (auto& x : input)
                    x = float(generation + 1);
                global_tensor<float, RowMajor<4, 8>> gm(input);
                SharedTile<Plane> tile;
                TLOAD<Plane, 8>(tile, gm);
                linx_cpu::publishSlot<3, 8>(tile);
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(pe));
        }
        for (unsigned generation = 0; generation < 8; ++generation) {
            SharedTile<Plane> tile;
            linx_cpu::acquireSlot<3>(tile);
            auto alias = tile;
            Plane result;
            TMOV_S2L_BROADCAST(result, alias);
            for (unsigned r = 0; r < 4; ++r)
                for (unsigned c = 0; c < 8; ++c)
                    check(linx_cpu::read(result, r, c) == float(generation + 1), "Shared slot generation lost");
        }
        linx_cpu::syncCore();
        if (pe == 0)
            check(linx_cpu::currentCore->sharedSlots.empty(), "consumed slot generations retained");
    });
    rejects([] {
        linx_cpu::runCore([] {
            SharedTile<Plane> tile;
            linx_cpu::acquireSlot<4>(tile);
        });
    });
}

static void viewsAndGpr()
{
    using Parent = VecTileM32<float, 32, 4, -1, 1>;
    using Fragment = VecTileM32<float, 32, 1, -1, 1>;
    Parent parent(17);
    TEXPANDS(parent, 2.0f);
    auto prefix = TREDUCEPREFIXVIEW<Fragment>(parent);
    Fragment zero(17), out(17);
    TEXPANDS(zero, 0.0f);
    TADD(out, zero, prefix);
    for (unsigned r = 0; r < 17; ++r)
        check(linx_cpu::read(out, r, 0) == 2.0f, "dynamic reduction-prefix row count");
    Tile<Location::Vec, float, 1, 1, BLayout::CubeM32, -1, 1> physical(17);
    TEXPANDS(physical, 7.0f);
    check(linx_cpu::read(physical, 16, 0) == 7, "CUBE physical capacity");
    decltype(physical) tooLarge(33);
    rejects([&] { TEXPANDS(tooLarge, 1.0f); });

    uint64_t input[32];
    for (unsigned i = 0; i < 32; ++i)
        input[i] = (uint64_t(i) << 40) + 7;
    ElementTile<uint64_t, 32> a, b;
    ElementTile<int32_t, 32> offsets, yes, no;
    linx_cpu::produce(offsets, 1, 32, [](auto r, auto) { return r == 31 ? uint64_t{0x7fffffff} : r * 8; });
    linx_cpu::gatherGpr(a, input, offsets, 0x7fffffff);
    linx_cpu::copy(b, a);
    check(linx_cpu::compareGpr<CmpMode::EQ>(a, b) == 0xffffffff, "64-bit GPR comparison");
    check(
        linx_cpu::readBits(a, 30, 0) == input[30] && linx_cpu::readBits(a, 31, 0) == 0,
        "masked gather touched inactive address");
    TEXPANDS(yes, 1);
    TEXPANDS(no, 2);
    linx_cpu::selectGpr(no, 0xaaaaaaaa, yes, no);
    for (unsigned i = 0; i < 32; ++i)
        check(linx_cpu::read(no, i, 0) == (i % 2 ? 1 : 2), "aliased GPR select");
}

static void tileArrayAndScalars()
{
    using Fragment = Tile<Location::Vec, float, 4, 8>;
    using Parent = Tile<Location::Vec, float, 8, 16>;
    region::TileArray<Fragment, 2, 2> array;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            Fragment input;
            linx_cpu::produce(input, 8, 4, [&](auto r, auto c) {
                return linx_cpu::scalarBits(float((i * 2 + j) * 100 + r * 10 + c));
            });
            auto slot = array[i][j];
            TCVT(slot, input);
        }
    auto result = TASSEMBLY<Parent>(std::move(array));
    for (unsigned r = 0; r < 8; ++r)
        for (unsigned c = 0; c < 16; ++c)
            check(
                linx_cpu::read(result, r, c) == float((r / 4 * 2 + c / 8) * 100 + r % 4 * 10 + c % 8),
                "TileArray rectangular assembly");
    check(__bf16(1.00390625f).data == 0x3f80, "BF16 ties to even low");
    check(__bf16(1.01171875f).data == 0x3f82, "BF16 ties to even high");
    check(__bf16(-0.0f).data == 0x8000, "BF16 signed zero");
    check(float(__bf16(3)) == 3.0f, "BF16 integer conversion");
    check(std::isnan(float(__bf16(std::numeric_limits<float>::quiet_NaN()))), "BF16 NaN");
}
int main()
{
    launchAndMemory();
    slotGenerations();
    viewsAndGpr();
    tileArrayAndScalars();
}
#endif
