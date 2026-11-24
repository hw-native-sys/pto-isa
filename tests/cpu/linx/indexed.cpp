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
#include <cstdint>
using namespace pto;
template <typename T>
using V = Tile<Location::Vec, T, 1, 32>;
template <typename T>
using M = global_tensor<T, RowMajor<1, 32>>;
template <bool Bounded = true>
void gatherAtomic(unsigned operation, uint32_t* memory, int32_t* index, uint32_t* operand, uint32_t* observed)
{
    V<int32_t> i;
    V<uint32_t> v, old;
    M<int32_t> mi(index);
    M<uint32_t> mv(operand), mo(observed);
    TLOAD(i, mi);
    TLOAD(v, mv);
    switch (operation) {
        case 0:
            if constexpr (Bounded)
                MGATHER_EXCH(old, reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MGATHER_EXCH(old, reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 1:
            if constexpr (Bounded)
                MGATHER_MAX(old, reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MGATHER_MAX(old, reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 2:
            if constexpr (Bounded)
                MGATHER_MIN(old, reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MGATHER_MIN(old, reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 3:
            if constexpr (Bounded)
                MGATHER_ADD(old, reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MGATHER_ADD(old, reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 4:
            if constexpr (Bounded)
                MGATHER_INC(old, reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MGATHER_INC(old, reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 5:
            if constexpr (Bounded)
                MGATHER_DEC(old, reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MGATHER_DEC(old, reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 6:
            if constexpr (Bounded)
                MGATHER_AND(old, reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MGATHER_AND(old, reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 7:
            if constexpr (Bounded)
                MGATHER_OR(old, reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MGATHER_OR(old, reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 8:
            if constexpr (Bounded)
                MGATHER_XOR(old, reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MGATHER_XOR(old, reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
    }
    TSTORE(mo, old);
}
template <bool Bounded = true>
void scatterAtomic(unsigned operation, uint32_t* memory, int32_t* index, uint32_t* operand)
{
    V<int32_t> i;
    V<uint32_t> v;
    M<int32_t> mi(index);
    M<uint32_t> mv(operand);
    TLOAD(i, mi);
    TLOAD(v, mv);
    switch (operation) {
        case 0:
            if constexpr (Bounded)
                MSCATTER_MAX(reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MSCATTER_MAX(reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 1:
            if constexpr (Bounded)
                MSCATTER_MIN(reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MSCATTER_MIN(reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 2:
            if constexpr (Bounded)
                MSCATTER_ADD(reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MSCATTER_ADD(reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 3:
            if constexpr (Bounded)
                MSCATTER_INC(reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MSCATTER_INC(reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 4:
            if constexpr (Bounded)
                MSCATTER_DEC(reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MSCATTER_DEC(reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 5:
            if constexpr (Bounded)
                MSCATTER_AND(reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MSCATTER_AND(reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 6:
            if constexpr (Bounded)
                MSCATTER_OR(reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MSCATTER_OR(reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 7:
            if constexpr (Bounded)
                MSCATTER_XOR(reinterpret_cast<uint64_t>(memory), i, v, 32, IndexedByteRange{0, 124}, 1);
            else
                MSCATTER_XOR(reinterpret_cast<uint64_t>(memory), i, v, 32, 1);
            break;
        case 8:
            if constexpr (Bounded)
                MSCATTER_POPC(reinterpret_cast<uint64_t>(memory), i, 32, IndexedByteRange{0, 124}, 1);
            else
                MSCATTER_POPC(reinterpret_cast<uint64_t>(memory), i, 32, 1);
            break;
    }
}
void compareExchange(uint32_t* memory, int32_t* index, uint32_t* expected, uint32_t* replacement, uint32_t* observed)
{
    V<int32_t> i;
    V<uint32_t> e, v, old;
    M<int32_t> mi(index);
    M<uint32_t> me(expected), mv(replacement), mo(observed);
    TLOAD(i, mi);
    TLOAD(e, me);
    TLOAD(v, mv);
    MGATHER_CAS(old, reinterpret_cast<uint64_t>(memory), i, e, v, 32);
    TSTORE(mo, old);
}
void regularIndexed(uint32_t* memory, int32_t* index, uint8_t* predicates, uint32_t* output)
{
    V<int32_t> i;
    V<uint32_t> v;
    Tile<Location::Vec, uint8_t, 1, 128, BLayout::RowMajor, 1, 32> mask;
    M<int32_t> mi(index);
    M<uint8_t> mp(predicates);
    M<uint32_t> src(memory), dst(output);
    TLOAD(i, mi);
    MGATHER(v, src, i, IndexedByteRange{-64, 60});
    TSTORE(dst, v);
    TLOAD(mask, mp);
    MGATHER_MASK(v, src, i, mask, IndexedByteRange{-64, 60});
    M<uint32_t> masked(output + 32);
    TSTORE(masked, v);
    M<uint32_t> scattered(output + 80);
    MSCATTER_MASK(scattered, v, i, mask, IndexedByteRange{-64, 60});
}
void atomicStore(unsigned operation, uint32_t* memory, uint32_t* operand)
{
    V<uint32_t> v;
    M<uint32_t> gm(memory), mv(operand);
    TLOAD(v, mv);
    switch (operation) {
        case 0:
            TSTORE_ATOMIC_MAX(gm, v);
            break;
        case 1:
            TSTORE_ATOMIC_MIN(gm, v);
            break;
        case 2:
            TSTORE_ATOMIC_ADD(gm, v);
            break;
        case 3:
            TSTORE_ATOMIC_INC(gm, v);
            break;
        case 4:
            TSTORE_ATOMIC_DEC(gm, v);
            break;
        case 5:
            TSTORE_ATOMIC_AND(gm, v);
            break;
        case 6:
            TSTORE_ATOMIC_OR(gm, v);
            break;
        case 7:
            TSTORE_ATOMIC_XOR(gm, v);
            break;
    }
}
void predicateAtomic(uint32_t* memory, int64_t* index, uint32_t* operand, uint32_t* output)
{
    using C = VecTileM32<uint32_t, 32, 4>;
    using I = VecTileM32<int64_t, 32, 4>;
    global_tensor<uint32_t, RowMajor<32, 4>> gm(memory), mv(operand), out(output);
    global_tensor<int64_t, RowMajor<32, 4>> mi(index);
    C v, result;
    I offsets;
    TLOAD(v, mv);
    TLOAD(offsets, mi);
    auto low = predicate::compare_scalar<CmpMode::GE>(v, 32u);
    auto high = predicate::compare_scalar<CmpMode::LT>(v, 96u);
    auto middle = predicate::logical<6>(low, high);
    predicate::atomic_add(gm, offsets, v, middle, IndexedByteRange{0, 508});
    predicate::select_scalar(result, middle, v, 99u);
    TSTORE(out, result);
}
// Predicate storage consists of complete U8 CELL columns, independently of
// the numeric basis dtype. Exercise the native private-carrier operations.
void predicateRegions(uint32_t* input, uint32_t* output)
{
    using T = VecTileM32<uint32_t, 32, 8>;
    T a, d;
    global_tensor<uint32_t, RowMajor<32, 8>> src(input), out(output);
    TLOAD(a, src);
    auto active = predicate::compare_scalar<CmpMode::GE>(a, 128u);
    auto left = predicate::subview<0, 4>(active);
    auto right = predicate::subview<1, 4>(active);
    auto leftLogical = predicate::logical_or(left, left);
    auto rightLogical = predicate::logical_and(right, right);
    auto rebuilt = predicate::concatenate(leftLogical, rightLogical);
    predicate::select_scalar(d, rebuilt, a, 999u);
    TSTORE(out, d);
}
#if defined(__CPU_SIM) && !defined(LINX_TEST_KERNEL_ONLY)
#include <array>
#include <iostream>
#include <stdexcept>
static void check(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}
static uint32_t expected(unsigned op, uint32_t old, uint32_t value)
{
    switch (op) {
        case 0:
            return value;
        case 1:
            return old > value ? old : value;
        case 2:
            return old < value ? old : value;
        case 3:
            return old + value;
        case 4:
            return old >= value ? 0 : old + 1;
        case 5:
            return old == 0 || old > value ? value : old - 1;
        case 6:
            return old & value;
        case 7:
            return old | value;
        default:
            return old ^ value;
    }
}
int main()
{
    std::array<uint32_t, 128> memory{}, operand{}, observed{}, compare{};
    std::array<int32_t, 128> indices{};
    for (unsigned i = 0; i < 32; ++i) {
        indices[i] = i * 4;
        operand[i] = i % 7;
        compare[i] = i % 2 ? i : 999;
    }
    for (unsigned op = 0; op < 9; ++op) {
        for (unsigned i = 0; i < 32; ++i)
            memory[i] = i;
        gatherAtomic(op, memory.data(), indices.data(), operand.data(), observed.data());
        for (unsigned i = 0; i < 32; ++i) {
            check(observed[i] == i, "atomic observed old value");
            check(memory[i] == expected(op, i, operand[i]), "MGATHER atomic operation");
        }
    }
    for (unsigned op = 0; op < 9; ++op) {
        for (unsigned i = 0; i < 32; ++i)
            memory[i] = i;
        scatterAtomic(op, memory.data(), indices.data(), operand.data());
        for (unsigned i = 0; i < 32; ++i)
            check(memory[i] == (op == 8 ? i + 1 : expected(op + 1, i, operand[i])), "MSCATTER atomic reduction");
    }
    for (unsigned op = 0; op < 8; ++op) {
        for (unsigned i = 0; i < 32; ++i)
            memory[i] = i;
        atomicStore(op, memory.data(), operand.data());
        for (unsigned i = 0; i < 32; ++i)
            check(memory[i] == expected(op + 1, i, operand[i]), "TSTORE atomic operation");
    }
    for (unsigned i = 0; i < 32; ++i)
        memory[i] = i;
    compareExchange(memory.data(), indices.data(), compare.data(), operand.data(), observed.data());
    for (unsigned i = 0; i < 32; ++i) {
        check(observed[i] == i, "CAS old value");
        check(memory[i] == (i % 2 ? operand[i] : i), "CAS compare and replacement");
    }
    std::array<uint8_t, 32> maskBytes{};
    for (unsigned i = 0; i < 32; ++i) {
        memory[i] = i;
        indices[i] = (int(i) - 16) * 4;
        maskBytes[i] = i >= 16;
    }
    observed.fill(93);
    regularIndexed(memory.data() + 16, indices.data(), maskBytes.data(), observed.data());
    for (unsigned i = 0; i < 32; ++i) {
        check(observed[i] == i, "signed byte displacement");
        check(observed[32 + i] == (i >= 16 ? i : 0), "masked gather inactive lanes");
        check(observed[64 + i] == (i >= 16 ? i : 93), "masked scatter preserves inactive lanes");
    }
    maskBytes[0] = 2;
    bool rejected = false;
    try {
        regularIndexed(memory.data() + 16, indices.data(), maskBytes.data(), observed.data());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "indexed masks reject noncanonical predicate bytes");
    for (unsigned i = 0; i < 128; ++i) {
        memory[i] = 1;
        operand[i] = i;
        indices[i] = i * 4;
    }
    std::array<int64_t, 128> wideIndices{};
    for (unsigned i = 0; i < 128; ++i)
        wideIndices[i] = indices[i];
    predicateAtomic(memory.data(), wideIndices.data(), operand.data(), observed.data());
    for (unsigned i = 0; i < 128; ++i) {
        bool active = i >= 32 && i < 96;
        check(memory[i] == (active ? i + 1 : 1), "Predicate atomic add");
        check(observed[i] == (active ? i : 99), "Predicate logical/select");
    }
    std::array<uint32_t, 256> predicateInput{}, predicateOutput{};
    for (unsigned i = 0; i < 256; ++i)
        predicateInput[i] = i;
    predicateRegions(predicateInput.data(), predicateOutput.data());
    for (unsigned i = 0; i < 256; ++i)
        check(predicateOutput[i] == (i >= 128 ? i : 999), "Predicate subview and assembly preserve CELL columns");
    std::cout << "Linx indexed memory, atomic, and Predicate execution checks passed\n";
}
#endif
