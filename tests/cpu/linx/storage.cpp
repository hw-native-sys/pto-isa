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
#include <array>
#include <cstdint>
using namespace pto;
template <typename T, int R, int C>
using Mem = global_tensor<T, RowMajor<R, C>>;
using Plane = Tile<Location::Vec, float, 4, 32>;
using Cube = VecTileM32<float, 32, 4>;
void sharedLoad(float* input, float* output)
{
    Mem<float, 4, 32> src(input + get_thread_idx() * 128), out(output + get_thread_idx() * 128);
    // A singleton issuer publishes a complete parent. Multiple issuers need
    // explicit B.ASSEMBLE ranges under the current native contract.
    auto shared = TLOAD<Plane, 8>(src);
    Plane result;
    TMOV_S2L_BROADCAST(result, shared);
    TSTORE(out, result);
}
void sharedSingleProducer(float* input, float* output)
{
    if (get_thread_idx() != 3)
        return;
    Mem<float, 4, 32> src(input + 384), first(output), second(output + 128);
    auto returned = TLOAD<Plane, 1>(src);
    Plane local;
    TMOV_S2L_BROADCAST<1>(local, returned);
    TSTORE(first, local);
    SharedTile<Plane> destination;
    TLOAD<Plane, 1>(destination, src);
    TMOV_S2L_BROADCAST<1>(local, destination);
    TSTORE(second, local);
}
// Native Clang accepts this API but its assembler rejects TSTORE.SPART in the
// pinned upstream too. The runner includes it in device syntax checks; define
// LINX_TEST_TSTORE_PART to reproduce the assembler limitation explicitly.
#if defined(__CPU_SIM) || defined(LINX_TEST_TSTORE_PART)
void sharedMaskedStore(float* input, float* output)
{
    Mem<float, 4, 32> src(input + get_thread_idx() * 128), out(output + get_thread_idx() * 128);
    auto shared = TLOAD<Plane>(src);
    TSTORE_PART<1>(out, shared);
}
#endif
void sharedND(float* input, float* output)
{
    Cube a, b;
    Mem<float, 32, 4> in(input + get_thread_idx() * 128), out(output + get_thread_idx() * 128);
    TLOAD(a, in);
    SharedTile<Tile<Location::Vec, float, 128, 4>> parent;
    auto first = range::assemble<4, 0>(parent);
    TMOV_CUBE_TO_SHARED_ND(first, a);
    auto second = range::assemble_middle<4, 4>(parent);
    TMOV_CUBE_TO_SHARED_ND(second, a);
    auto third = range::assemble_middle<4, 8>(parent);
    TMOV_CUBE_TO_SHARED_ND(third, a);
    auto last = range::assemble_last<4, 12>(parent);
    TMOV_CUBE_TO_SHARED_ND(last, a);
    auto view = range::shared_nd_subview<4>(parent, ((get_thread_idx() + 1) % 4) * 4);
    TMOV_SHARED_ND_TO_CUBE(b, view);
    TSTORE(out, b);
}
void packedMove(__uint4x2* input, __uint4x2* output)
{
    VecTileM16<__uint4x2, 16, 32> tile;
    Mem<__uint4x2, 16, 32> in(input), out(output);
    TLOAD(tile, in);
    TSTORE(out, tile);
}
void localAssembly(float* input, float* output)
{
    using Parent = VecTileM32<float, 32, 8>;
    using Writer = VecTileM32<float, 32, 4>;
    LocalCubeAssembly<Parent, Writer, Writer> assembly;
    Mem<float, 32, 4> first(input), second(input + 128);
    auto writer0 = assembly.writer<0>();
    TLOAD(writer0, first);
    auto writer1 = assembly.writer<1>();
    TLOAD_ASS(writer1, second);
    auto result = assembly.finish();
    Mem<float, 32, 8> out(output);
    TSTORE(out, result);
}
void rearrange(uint32_t* input, uint32_t* output)
{
    VecTileM32<uint32_t, 32, 1> a, b, result, controls;
    VecTileM32<uint8_t, 32, 4> indices;
    Mem<uint32_t, 32, 1> in(input), out(output);
    TLOAD(a, in);
    TEXPANDS(b, 0x77665544u);
    TPACK(result, a, b, 0x0202);
    TSTORE(out, result);
    TUNPACK(result, a, 0x0201);
    Mem<uint32_t, 32, 1> unpacked(output + 32);
    TSTORE(unpacked, result);
    TEXPANDS(controls, 1u);
    TSHUF(result, a, controls, 0x000400);
    Mem<uint32_t, 32, 1> shuffled(output + 64);
    TSTORE(shuffled, result);
    TEXPANDS(indices, uint8_t(6));
    TPERMUTE(result, a, b, indices);
    Mem<uint32_t, 32, 1> permuted(output + 96);
    TSTORE(permuted, result);
}
void makeBits(uint8_t* output)
{
    VecTileM32<uint8_t, 32, 4> tile;
    TGPR2T(tile, 0x00000000ffffffffULL, 0xffffffffffffffffULL, 0, 0);
    Mem<uint8_t, 32, 4> out(output);
    TSTORE(out, tile);
}
#if defined(__CPU_SIM) && !defined(LINX_TEST_KERNEL_ONLY)
#include <atomic>
#include <iostream>
#include <stdexcept>
static void check(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}
int main()
{
    std::array<float, 512> input{}, output{};
    for (unsigned i = 0; i < input.size(); ++i)
        input[i] = i;
    linx_cpu::runCore([&] { sharedLoad(input.data(), output.data()); });
    for (unsigned pe = 0; pe < 4; ++pe)
        for (unsigned i = 0; i < 128; ++i)
            check(output[pe * 128 + i] == input[i], "Shared TLOAD singleton publication");
    output.fill(-1);
    linx_cpu::runCore([&] { sharedSingleProducer(input.data(), output.data()); });
    for (unsigned i = 0; i < 256; ++i)
        check(output[i] == input[384 + i % 128], "Shared singleton must not wait for nonselected PEs");
    linx_cpu::runCore([&] {
        if (get_thread_idx() == 0) {
            Mem<float, 4, 32> unused(nullptr);
            auto ignored = TLOAD<Plane, 1>(unused);
            (void)ignored;
        }
    });
    std::atomic<unsigned> nonselectedReturned{0};
    linx_cpu::runCore([&] {
        const auto pe = get_thread_idx();
        if (pe == 3)
            while (nonselectedReturned.load() != 3)
                std::this_thread::yield();
        Mem<float, 4, 32> src(input.data() + pe * 128), out(output.data() + pe * 128);
        auto pending = TLOAD<Plane, 1>(src);
        if (pe != 3)
            ++nonselectedReturned;
        TSTORE(out, pending);
    });
    for (unsigned i = 0; i < 512; ++i)
        check(output[i] == input[384 + i % 128], "late singleton publication reaches every Shared consumer");
    output.fill(-1);
    linx_cpu::runCore([&] { sharedMaskedStore(input.data(), output.data()); });
    for (unsigned pe = 0; pe < 4; ++pe)
        for (unsigned i = 0; i < 128; ++i)
            check(
                output[pe * 128 + i] == (pe == 3 ? input[(i / 32) * 128 + i] : -1),
                "TSTORE_PART selects consumers without selecting quarters");
    bool missingParticipant = false;
    try {
        linx_cpu::runCore([&] {
            if (get_thread_idx() == 3) {
                Mem<float, 4, 32> src(input.data());
                auto incomplete = TLOAD<Plane>(src);
                (void)incomplete;
            }
        });
    } catch (const std::invalid_argument&) {
        missingParticipant = true;
    }
    check(missingParticipant, "Shared publication must reject a missing selected producer");
    linx_cpu::runCore([&] { sharedND(input.data(), output.data()); });
    for (unsigned pe = 0; pe < 4; ++pe)
        for (unsigned i = 0; i < 128; ++i)
            check(output[pe * 128 + i] == input[((pe + 1) % 4) * 128 + i], "Shared ND assembly publication/subview");
    std::array<__uint4x2, 256> packed{}, unpacked{};
    for (unsigned i = 0; i < packed.size(); ++i)
        packed[i].data = i;
    packedMove(packed.data(), unpacked.data());
    check(std::memcmp(packed.data(), unpacked.data(), 256) == 0, "packed M16 permutation and GM addressing");
    localAssembly(input.data(), output.data());
    for (unsigned r = 0; r < 32; ++r)
        for (unsigned c = 0; c < 8; ++c)
            check(output[r * 8 + c] == input[(c / 4) * 128 + r * 4 + c % 4], "Local M32 ordered assembly");
    Plane local, result;
    TEXPANDS(local, 3.f);
    auto unpublished = TMOV_L2S_INSERT<8>(local);
    bool rejected = false;
    try {
        TMOV_S2L_BROADCAST(result, unpublished);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "Shared read before publication must fail");
    TMOV_L2S_PUBLISH<8>(unpublished, local);
    TMOV_S2L_BROADCAST(result, unpublished);
    std::array<uint32_t, 32> words{};
    std::array<uint32_t, 128> rearranged{};
    for (unsigned i = 0; i < 32; ++i)
        words[i] = 0xaabb0000u + i;
    rearrange(words.data(), rearranged.data());
    for (unsigned i = 0; i < 32; ++i) {
        check(rearranged[i] == (0x55440000u | i), "TPACK word prefix");
        check(rearranged[32 + i] == 0xbb00, "TUNPACK offset/count");
        check(rearranged[64 + i] == words[i ? i - 1 : 0], "TSHUF up with self boundary");
        check(rearranged[96 + i] == 0x66666666, "TPERMUTE two-source byte selector");
    }
    std::array<uint8_t, 128> planes{};
    makeBits(planes.data());
    for (unsigned r = 0; r < 32; ++r)
        for (unsigned c = 0; c < 4; ++c)
            check(planes[r * 4 + c] == (c == 0 ? 13 : 0), "TGPR2T bit planes");
    {
        using P = VecTileM32<uint32_t, 32, 2>;
        P parent;
        VecTileM32<uint32_t, 32, 1> a, b;
        TEXPANDS(a, 3u);
        TEXPANDS(b, 2u);
        range::Assemble<P, 2, true, false, 0, 0, 1> first(parent);
        TPACK(first, a, b, 0x0202);
        auto last = range::assemble_last<1, 1>(parent);
        TPACK_ASS(last, a, b, 0x0202);
        std::array<uint32_t, 64> packedWords{};
        Mem<uint32_t, 32, 2> out(packedWords.data());
        TSTORE(out, parent);
        for (auto word : packedWords)
            check(word == 0x00020003u, "TPACK_ASS must use writer width, not parent width");
    }
    std::cout << "Linx storage, assembly, and collective execution checks passed\n";
}
#endif
