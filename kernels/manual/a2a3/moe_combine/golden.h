/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_COMBINE_GOLDEN_H_
#define MOE_COMBINE_GOLDEN_H_

#include "args.h"
#include "layout.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace moe_combine {

struct HostInputData {
    std::vector<float> inputA;
    std::vector<int32_t> expertIdx;
    std::vector<float> probs;
};

struct CpuGoldenData {
    std::vector<float> dispatchedA;
    std::vector<float> expertOutput;
    std::vector<float> outputC;
    std::vector<float> ptrD;
    std::vector<float> packedA;
    std::vector<int32_t> localTokenPerExpert;
    std::vector<int32_t> peerTokenPerExpert;
    std::vector<int32_t> expandedRowIdx;
    std::vector<int32_t> cumsumPerExpert;
    std::vector<int32_t> dispatchOffset;
    std::vector<int32_t> prevSumBeforeRank;
    std::vector<uint32_t> ownerRows;
    uint64_t totalRoutes = 0;
    uint64_t invalidRoutes = 0;
};

struct CompareResult {
    uint64_t elementCount;
    uint64_t mismatchCount;
    uint64_t firstMismatchIndex;
    float actual;
    float expected;
};

namespace golden_detail {

struct RouteRef {
    uint32_t src = 0;
    uint32_t token = 0;
    uint32_t slot = 0;
    uint32_t expert = 0;
    uint32_t packedRow = 0;
};

inline std::string RankFile(const MoeCombineArgs &args, uint32_t rank, const char *name)
{
    return args.dataDir + "/rank_" + std::to_string(rank) + "_" + name + ".bin";
}

inline void EnsureDataDir(const std::string &path)
{
    if (path.empty()) {
        return;
    }
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("mkdir failed for " + path + ": " + std::strerror(errno));
    }
}

template <typename T>
inline void WriteBinary(const std::string &path, const std::vector<T> &data)
{
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    os.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!os.good()) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

template <typename T>
inline std::vector<T> ReadBinary(const std::string &path, size_t elementCount)
{
    std::vector<T> data(elementCount);
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    is.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (is.gcount() != static_cast<std::streamsize>(data.size() * sizeof(T))) {
        throw std::runtime_error("input file size mismatch: " + path);
    }
    return data;
}

inline uint16_t FloatToHalf(float value)
{
    union {
        float f;
        uint32_t u;
    } bits{};
    bits.f = value;
    uint32_t sign = (bits.u >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits.u >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = bits.u & 0x007FFFFFU;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

inline float HalfToFloat(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exp = (value >> 10) & 0x1FU;
    uint32_t mant = value & 0x03FFU;
    uint32_t out = 0;
    if (exp == 0) {
        out = sign;
    } else if (exp == 31) {
        out = sign | 0x7F800000U | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    union {
        uint32_t u;
        float f;
    } bits{out};
    return bits.f;
}

inline std::vector<uint16_t> FloatVectorToHalf(const std::vector<float> &src)
{
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = FloatToHalf(src[i]);
    }
    return dst;
}

inline std::vector<float> HalfVectorToFloat(const std::vector<uint16_t> &src)
{
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = HalfToFloat(src[i]);
    }
    return dst;
}

inline HostInputData GenerateDeterministicInputs(const MoeCombineArgs &args, uint32_t rank)
{
    const MoeCombineShape &shape = args.shape;
    HostInputData data;
    data.inputA.resize(static_cast<size_t>(shape.m) * shape.k);
    data.expertIdx.resize(static_cast<size_t>(shape.m) * shape.topK);
    data.probs.resize(static_cast<size_t>(shape.m) * shape.topK);

    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t col = 0; col < shape.k; ++col) {
            uint32_t mixed = args.runtime.seed + rank * 131U + token * 17U + col * 3U;
            data.inputA[static_cast<size_t>(token) * shape.k + col] = static_cast<float>(mixed % 251U) / 32.0f;
        }
        float probSum = 0.0f;
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t flat = token * shape.topK + slot;
            data.expertIdx[flat] = static_cast<int32_t>((flat + rank) % shape.expertNum);
            float prob = static_cast<float>((args.runtime.seed + rank + flat) % 7U + 1U);
            data.probs[flat] = prob;
            probSum += prob;
        }
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t flat = token * shape.topK + slot;
            data.probs[flat] /= probSum;
        }
    }
    return data;
}

inline HostInputData LoadInputs(const MoeCombineArgs &args, uint32_t rank)
{
    const MoeCombineShape &shape = args.shape;
    size_t inputElems = static_cast<size_t>(shape.m) * shape.k;
    size_t routeElems = static_cast<size_t>(shape.m) * shape.topK;
    HostInputData data;
    data.inputA = HalfVectorToFloat(ReadBinary<uint16_t>(RankFile(args, rank, "inputA"), inputElems));
    data.expertIdx = ReadBinary<int32_t>(RankFile(args, rank, "expertIdx"), routeElems);
    data.probs = ReadBinary<float>(RankFile(args, rank, "probs"), routeElems);
    return data;
}

inline void WriteInputs(const MoeCombineArgs &args, uint32_t rank, const HostInputData &data)
{
    EnsureDataDir(args.dataDir);
    WriteBinary(RankFile(args, rank, "inputA"), FloatVectorToHalf(data.inputA));
    WriteBinary(RankFile(args, rank, "expertIdx"), data.expertIdx);
    WriteBinary(RankFile(args, rank, "probs"), data.probs);
}

inline std::vector<HostInputData> LoadOrGenerateWorldInputs(const MoeCombineArgs &args)
{
    std::vector<HostInputData> world(args.shape.ep);
    for (uint32_t rank = 0; rank < args.shape.ep; ++rank) {
        world[rank] = args.runtime.genData != 0 ? GenerateDeterministicInputs(args, rank) : LoadInputs(args, rank);
    }
    return world;
}

inline void WriteDebugFiles(const MoeCombineArgs &args, uint32_t rank, const CpuGoldenData &golden)
{
    if (args.runtime.debug == 0) {
        return;
    }
    EnsureDataDir(args.dataDir);
    WriteBinary(RankFile(args, rank, "localTokenPerExpert"), golden.localTokenPerExpert);
    WriteBinary(RankFile(args, rank, "peerTokenPerExpert"), golden.peerTokenPerExpert);
    WriteBinary(RankFile(args, rank, "cumsumPerExpert"), golden.cumsumPerExpert);
    WriteBinary(RankFile(args, rank, "expandedRowIdx"), golden.expandedRowIdx);
    WriteBinary(RankFile(args, rank, "packedA_head"), FloatVectorToHalf(golden.packedA));
    WriteBinary(RankFile(args, rank, "dispatchedA_head"), FloatVectorToHalf(golden.dispatchedA));
    WriteBinary(RankFile(args, rank, "ptrD_head"), FloatVectorToHalf(golden.ptrD));
}

inline bool IsValidExpert(int32_t expert, uint32_t expertNum)
{
    return expert >= 0 && static_cast<uint32_t>(expert) < expertNum;
}

inline void CountSourceRoutes(const MoeCombineShape &shape, const HostInputData &input, uint32_t src,
                              uint32_t expertNumPadded, std::vector<uint32_t> *localTokenPerExpert,
                              std::vector<int32_t> *peerTokenPerExpert, uint64_t *totalRoutes, uint64_t *invalidRoutes)
{
    localTokenPerExpert->assign(shape.expertNum, 0);
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            ++(*totalRoutes);
            size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
            int32_t expert = input.expertIdx[routeIndex];
            if (!IsValidExpert(expert, shape.expertNum)) {
                ++(*invalidRoutes);
                continue;
            }
            ++(*localTokenPerExpert)[static_cast<uint32_t>(expert)];
        }
    }
    for (uint32_t expert = 0; expert < shape.expertNum; ++expert) {
        (*peerTokenPerExpert)[static_cast<size_t>(src) * expertNumPadded + expert] =
            static_cast<int32_t>((*localTokenPerExpert)[expert]);
    }
}

inline std::vector<uint32_t> BuildExpertBase(const MoeCombineShape &shape,
                                             const std::vector<uint32_t> &localTokenPerExpert)
{
    std::vector<uint32_t> expertBase(shape.expertNum, 0);
    uint32_t running = 0;
    for (uint32_t expert = 0; expert < shape.expertNum; ++expert) {
        expertBase[expert] = running;
        running += localTokenPerExpert[expert];
    }
    return expertBase;
}

inline void CopyPackedTokenRow(const MoeCombineShape &shape, const HostInputData &input, uint32_t token,
                               uint32_t packedRow, std::vector<float> *packed)
{
    for (uint32_t col = 0; col < shape.k; ++col) {
        (*packed)[static_cast<size_t>(packedRow) * shape.k + col] =
            input.inputA[static_cast<size_t>(token) * shape.k + col];
    }
}

inline void FillSourceRoutes(const MoeCombineShape &shape, const HostInputData &input, uint32_t src,
                             const std::vector<uint32_t> &expertBase,
                             std::vector<std::vector<RouteRef>> *routesByExpert, std::vector<float> *packed,
                             std::vector<int32_t> *expanded)
{
    std::vector<uint32_t> cursor(shape.expertNum, 0);
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
            int32_t expert = input.expertIdx[routeIndex];
            if (!IsValidExpert(expert, shape.expertNum)) {
                continue;
            }
            uint32_t expertId = static_cast<uint32_t>(expert);
            uint32_t packedRow = expertBase[expertId] + cursor[expertId]++;
            (*expanded)[routeIndex] = static_cast<int32_t>(packedRow);
            (*routesByExpert)[expertId].push_back(RouteRef{src, token, slot, expertId, packedRow});
            CopyPackedTokenRow(shape, input, token, packedRow, packed);
        }
    }
}

inline void BuildRoutes(const MoeCombineArgs &args, const std::vector<HostInputData> &worldInputs,
                        std::vector<std::vector<std::vector<RouteRef>>> *routesBySrcExpert,
                        std::vector<std::vector<float>> *packedBySrc, std::vector<std::vector<int32_t>> *expandedBySrc,
                        std::vector<int32_t> *peerTokenPerExpert, uint64_t *totalRoutes, uint64_t *invalidRoutes)
{
    const MoeCombineShape &shape = args.shape;
    uint32_t expertNumPadded = static_cast<uint32_t>(ExpertNumPadded(shape));
    uint32_t expandedRows = shape.m * shape.topK;
    routesBySrcExpert->assign(shape.ep, std::vector<std::vector<RouteRef>>(shape.expertNum));
    packedBySrc->assign(shape.ep, std::vector<float>(static_cast<size_t>(expandedRows) * shape.k, 0.0f));
    expandedBySrc->assign(shape.ep, std::vector<int32_t>(expandedRows, -1));
    peerTokenPerExpert->assign(static_cast<size_t>(shape.ep) * expertNumPadded, 0);
    *totalRoutes = 0;
    *invalidRoutes = 0;

    for (uint32_t src = 0; src < shape.ep; ++src) {
        std::vector<uint32_t> localTokenPerExpert;
        CountSourceRoutes(shape, worldInputs[src], src, expertNumPadded, &localTokenPerExpert, peerTokenPerExpert,
                          totalRoutes, invalidRoutes);
        std::vector<uint32_t> expertBase = BuildExpertBase(shape, localTokenPerExpert);
        FillSourceRoutes(shape, worldInputs[src], src, expertBase, &(*routesBySrcExpert)[src], &(*packedBySrc)[src],
                         &(*expandedBySrc)[src]);
    }
}

} // namespace golden_detail

inline HostInputData GenerateOrLoadInputs(const MoeCombineArgs &args, uint32_t myRank)
{
    HostInputData inputs = args.runtime.genData != 0 ? golden_detail::GenerateDeterministicInputs(args, myRank) :
                                                       golden_detail::LoadInputs(args, myRank);
    if (args.runtime.genData != 0) {
        golden_detail::WriteInputs(args, myRank, inputs);
    }
    return inputs;
}

inline void GenerateAllInputFiles(const MoeCombineArgs &args)
{
    for (uint32_t rank = 0; rank < args.shape.ep; ++rank) {
        golden_detail::WriteInputs(args, rank, golden_detail::GenerateDeterministicInputs(args, rank));
    }
}

inline std::string RankBinaryFile(const MoeCombineArgs &args, uint32_t rank, const char *name)
{
    return golden_detail::RankFile(args, rank, name);
}

template <typename T>
inline void WriteBinaryFile(const std::string &path, const std::vector<T> &data)
{
    golden_detail::WriteBinary(path, data);
}

inline std::vector<uint16_t> FloatVectorToHalfBits(const std::vector<float> &src)
{
    return golden_detail::FloatVectorToHalf(src);
}

inline std::vector<float> HalfBitsToFloatVector(const std::vector<uint16_t> &src)
{
    return golden_detail::HalfVectorToFloat(src);
}

using RouteTable = std::vector<std::vector<std::vector<golden_detail::RouteRef>>>;

inline void InitGoldenLocalData(const MoeCombineShape &shape, uint32_t myRank, uint32_t expertNumPadded,
                                const std::vector<std::vector<float>> &packedBySrc,
                                const std::vector<std::vector<int32_t>> &expandedBySrc, CpuGoldenData *golden)
{
    golden->localTokenPerExpert.assign(expertNumPadded, 0);
    for (uint32_t expert = 0; expert < shape.expertNum; ++expert) {
        golden->localTokenPerExpert[expert] =
            golden->peerTokenPerExpert[static_cast<size_t>(myRank) * expertNumPadded + expert];
    }
    golden->expandedRowIdx = expandedBySrc[myRank];
    golden->packedA = packedBySrc[myRank];
}

inline void BuildCumsumPerExpert(const MoeCombineShape &shape, uint32_t expertNumPadded, CpuGoldenData *golden)
{
    golden->cumsumPerExpert.assign(static_cast<size_t>(shape.ep) * expertNumPadded, 0);
    for (uint32_t src = 0; src < shape.ep; ++src) {
        int32_t sum = 0;
        for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
            sum += golden->peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + expert];
            golden->cumsumPerExpert[static_cast<size_t>(src) * expertNumPadded + expert] = sum;
        }
    }
}

inline void BuildOwnerRows(const MoeCombineShape &shape, const RouteTable &routesBySrcExpert, CpuGoldenData *golden)
{
    golden->ownerRows.assign(shape.ep, 0);
    for (uint32_t owner = 0; owner < shape.ep; ++owner) {
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = owner * shape.expertPerRank + localExpert;
            for (uint32_t src = 0; src < shape.ep; ++src) {
                golden->ownerRows[owner] += routesBySrcExpert[src][globalExpert].size();
            }
        }
    }
}

inline void BuildDispatchPlan(const MoeCombineShape &shape, uint32_t myRank, const RouteTable &routesBySrcExpert,
                              CpuGoldenData *golden)
{
    golden->dispatchOffset.assign(shape.expertPerRank, 0);
    golden->prevSumBeforeRank.assign(static_cast<size_t>(shape.ep) * shape.expertPerRank, 0);
    uint32_t dispatchCursor = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        golden->dispatchOffset[localExpert] = static_cast<int32_t>(dispatchCursor);
        uint32_t beforeRank = 0;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            golden->prevSumBeforeRank[static_cast<size_t>(src) * shape.expertPerRank + localExpert] =
                static_cast<int32_t>(beforeRank);
            uint32_t rows = static_cast<uint32_t>(routesBySrcExpert[src][globalExpert].size());
            beforeRank += rows;
            dispatchCursor += rows;
        }
    }
    if (dispatchCursor > shape.maxOutputSize) {
        throw std::runtime_error("CPU golden dispatched rows exceed maxOutputSize");
    }
}

inline void CopyPackedToDispatchedRow(const MoeCombineShape &shape, uint32_t outRow, uint32_t packedRow,
                                      const std::vector<float> &packed, CpuGoldenData *golden)
{
    for (uint32_t col = 0; col < shape.k; ++col) {
        golden->dispatchedA[static_cast<size_t>(outRow) * shape.k + col] =
            packed[static_cast<size_t>(packedRow) * shape.k + col];
    }
}

inline void FillDispatchedA(const MoeCombineShape &shape, uint32_t myRank, const RouteTable &routesBySrcExpert,
                            const std::vector<std::vector<float>> &packedBySrc, CpuGoldenData *golden)
{
    golden->dispatchedA.assign(static_cast<size_t>(shape.maxOutputSize) * shape.k, 0.0f);
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            uint32_t dstStart =
                static_cast<uint32_t>(golden->dispatchOffset[localExpert]) +
                static_cast<uint32_t>(
                    golden->prevSumBeforeRank[static_cast<size_t>(src) * shape.expertPerRank + localExpert]);
            const auto &routes = routesBySrcExpert[src][globalExpert];
            for (uint32_t row = 0; row < routes.size(); ++row) {
                CopyPackedToDispatchedRow(shape, dstStart + row, routes[row].packedRow, packedBySrc[src], golden);
            }
        }
    }
    golden->expertOutput = golden->dispatchedA;
}

inline void WritePtrDRouteRow(const MoeCombineShape &shape, uint32_t myRank, uint32_t dst, uint32_t src,
                              uint32_t expertOutputRow, const golden_detail::RouteRef &route,
                              const std::vector<std::vector<int32_t>> &expandedBySrc,
                              const std::vector<std::vector<float>> &packedBySrc, const CpuGoldenData &golden,
                              std::vector<std::vector<float>> *ptrDBySrc)
{
    int32_t dstPackedRow = expandedBySrc[src][static_cast<size_t>(route.token) * shape.topK + route.slot];
    if (dstPackedRow < 0) {
        return;
    }
    for (uint32_t col = 0; col < shape.k; ++col) {
        float value = dst == myRank ? golden.expertOutput[static_cast<size_t>(expertOutputRow) * shape.k + col] :
                                      packedBySrc[src][static_cast<size_t>(route.packedRow) * shape.k + col];
        (*ptrDBySrc)[src][static_cast<size_t>(dstPackedRow) * shape.k + col] = value;
    }
}

inline void FillPtrDForDestination(const MoeCombineShape &shape, uint32_t myRank, uint32_t dst,
                                   const RouteTable &routesBySrcExpert,
                                   const std::vector<std::vector<int32_t>> &expandedBySrc,
                                   const std::vector<std::vector<float>> &packedBySrc, const CpuGoldenData &golden,
                                   std::vector<std::vector<float>> *ptrDBySrc)
{
    uint32_t dstDispatchCursor = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = dst * shape.expertPerRank + localExpert;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            const auto &routes = routesBySrcExpert[src][globalExpert];
            for (uint32_t row = 0; row < routes.size(); ++row) {
                WritePtrDRouteRow(shape, myRank, dst, src, dstDispatchCursor + row, routes[row], expandedBySrc,
                                  packedBySrc, golden, ptrDBySrc);
            }
            dstDispatchCursor += routes.size();
        }
    }
}

inline void BuildPtrD(const MoeCombineShape &shape, uint32_t myRank, const RouteTable &routesBySrcExpert,
                      const std::vector<std::vector<int32_t>> &expandedBySrc,
                      const std::vector<std::vector<float>> &packedBySrc, CpuGoldenData *golden)
{
    uint32_t expandedRows = shape.m * shape.topK;
    std::vector<std::vector<float>> ptrDBySrc(shape.ep,
                                              std::vector<float>(static_cast<size_t>(expandedRows) * shape.k, 0.0f));
    for (uint32_t dst = 0; dst < shape.ep; ++dst) {
        FillPtrDForDestination(shape, myRank, dst, routesBySrcExpert, expandedBySrc, packedBySrc, *golden, &ptrDBySrc);
    }
    golden->ptrD = ptrDBySrc[myRank];
}

inline void RestoreOutputC(const MoeCombineShape &shape, const HostInputData &localInput, CpuGoldenData *golden)
{
    golden->outputC.assign(static_cast<size_t>(shape.m) * shape.k, 0.0f);
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
            int32_t ptrDRow = golden->expandedRowIdx[routeIndex];
            if (ptrDRow < 0) {
                continue;
            }
            float prob = localInput.probs[routeIndex];
            for (uint32_t col = 0; col < shape.k; ++col) {
                golden->outputC[static_cast<size_t>(token) * shape.k + col] +=
                    prob * golden->ptrD[static_cast<size_t>(ptrDRow) * shape.k + col];
            }
        }
    }
}

inline void WriteGoldenOutputFiles(const MoeCombineArgs &args, uint32_t myRank, const CpuGoldenData &golden)
{
    golden_detail::EnsureDataDir(args.dataDir);
    golden_detail::WriteBinary(golden_detail::RankFile(args, myRank, "golden_outputC"),
                               golden_detail::FloatVectorToHalf(golden.outputC));
    golden_detail::WriteDebugFiles(args, myRank, golden);
}

CpuGoldenData ComputeCpuGolden(const MoeCombineArgs &args, const HostInputData &inputs, uint32_t myRank);

CompareResult CompareOutputs(const MoeCombineArgs &args, const CpuGoldenData &golden,
                             const std::vector<float> &actualOutputC, uint32_t myRank);

} // namespace moe_combine

#endif // MOE_COMBINE_GOLDEN_H_
