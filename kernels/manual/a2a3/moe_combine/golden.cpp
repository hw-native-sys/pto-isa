/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "golden.h"

#include <cmath>

namespace moe_combine {

CpuGoldenData ComputeCpuGolden(const MoeCombineArgs& args, const HostInputData& inputs, uint32_t myRank)
{
    (void)inputs;
    const MoeCombineShape& shape = args.shape;
    uint32_t expertNumPadded = static_cast<uint32_t>(ExpertNumPadded(shape));

    std::vector<HostInputData> worldInputs = golden_detail::LoadOrGenerateWorldInputs(args);
    RouteTable routesBySrcExpert;
    std::vector<std::vector<float>> packedBySrc;
    std::vector<std::vector<int32_t>> expandedBySrc;

    CpuGoldenData golden;
    golden_detail::BuildRoutes(
        args, worldInputs, &routesBySrcExpert, &packedBySrc, &expandedBySrc, &golden.peerTokenPerExpert,
        &golden.totalRoutes, &golden.invalidRoutes);

    InitGoldenLocalData(shape, myRank, expertNumPadded, packedBySrc, expandedBySrc, &golden);
    BuildCumsumPerExpert(shape, expertNumPadded, &golden);
    BuildOwnerRows(shape, routesBySrcExpert, &golden);
    BuildDispatchPlan(shape, myRank, routesBySrcExpert, &golden);
    FillDispatchedA(shape, myRank, routesBySrcExpert, packedBySrc, &golden);
    BuildPtrD(shape, myRank, routesBySrcExpert, expandedBySrc, packedBySrc, &golden);
    RestoreOutputC(shape, worldInputs[myRank], &golden);
    WriteGoldenOutputFiles(args, myRank, golden);
    return golden;
}

CompareResult CompareOutputs(
    const MoeCombineArgs& args, const CpuGoldenData& golden, const std::vector<float>& actualOutputC, uint32_t myRank)
{
    (void)myRank;
    CompareResult result{};
    result.elementCount = static_cast<uint64_t>(golden.outputC.size());
    result.firstMismatchIndex = result.elementCount;
    result.actual = 0.0f;
    result.expected = 0.0f;
    if (actualOutputC.size() != golden.outputC.size()) {
        result.mismatchCount = result.elementCount;
        result.firstMismatchIndex = 0;
        return result;
    }
    for (size_t i = 0; i < golden.outputC.size(); ++i) {
        float actual = actualOutputC[i];
        float expected = golden.outputC[i];
        float diff = std::fabs(actual - expected);
        float tol = static_cast<float>(args.atol + args.rtol * std::fabs(expected));
        if (diff > tol) {
            if (result.mismatchCount == 0) {
                result.firstMismatchIndex = i;
                result.actual = actual;
                result.expected = expected;
            }
            ++result.mismatchCount;
        }
    }
    return result;
}

} // namespace moe_combine
