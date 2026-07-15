/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_COMBINE_ARGS_H_
#define MOE_COMBINE_ARGS_H_

#include "common.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace moe_combine {

#ifndef CONFIG_MOE_COMBINE_MIX_AIC_BLOCKS
#define CONFIG_MOE_COMBINE_MIX_AIC_BLOCKS 20U
#endif

#ifndef CONFIG_MOE_COMBINE_MIX_AIV_RATIO
#define CONFIG_MOE_COMBINE_MIX_AIV_RATIO 2U
#endif

#ifndef CONFIG_MOE_COMBINE_AIV_NUM
#define CONFIG_MOE_COMBINE_AIV_NUM (CONFIG_MOE_COMBINE_MIX_AIC_BLOCKS * CONFIG_MOE_COMBINE_MIX_AIV_RATIO)
#endif

#ifndef CONFIG_MOE_COMBINE_MAX_AIV_NUM
#define CONFIG_MOE_COMBINE_MAX_AIV_NUM 128U
#endif

struct MoeCombineArgs {
    MoeCombineShape shape;
    MoeCombineRuntimeConfig runtime;
    std::string runMode = "npu";
    std::string socVersion = "Ascend950PR_958b";
    std::string mpiBin;
    std::string dataDir = "./out";
    double rtol = 1e-2;
    double atol = 1e-2;
    bool rankSet = false;
    bool nranksSet = false;
};

struct MoeCombineResourceConfig {
    uint32_t defaultAicBlocks;
    uint32_t defaultAivRatio;
    uint32_t defaultAivBlocks;
    uint32_t maxAivBlocks;
};

inline bool StartsWith(const std::string& value, const char* prefix) { return value.rfind(prefix, 0) == 0; }

inline bool IsA5Soc(const std::string& socVersion)
{
    return socVersion.empty() || socVersion == "Ascend950" || StartsWith(socVersion, "Ascend950DT_") ||
           StartsWith(socVersion, "Ascend950PR_");
}

inline MoeCombineResourceConfig GetResourceConfig(const std::string& socVersion)
{
    if (!IsA5Soc(socVersion)) {
        throw std::runtime_error("unsupported moe_combine A5 resource config for soc: " + socVersion);
    }
    return MoeCombineResourceConfig{
        CONFIG_MOE_COMBINE_MIX_AIC_BLOCKS, CONFIG_MOE_COMBINE_MIX_AIV_RATIO, CONFIG_MOE_COMBINE_AIV_NUM,
        CONFIG_MOE_COMBINE_MAX_AIV_NUM};
}

inline uint32_t ChooseDefaultAivBlocks(const std::string& socVersion, const MoeCombineShape&)
{
    return GetResourceConfig(socVersion).defaultAivBlocks;
}

inline uint32_t ParseU32(const std::string& value, const char* name)
{
    size_t parsed = 0;
    unsigned long out = std::stoul(value, &parsed, 10);
    if (parsed != value.size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + name + ": " + value);
    }
    return static_cast<uint32_t>(out);
}

inline double ParseDouble(const std::string& value, const char* name)
{
    size_t parsed = 0;
    double out = std::stod(value, &parsed);
    if (parsed != value.size()) {
        throw std::invalid_argument(std::string("invalid float for ") + name + ": " + value);
    }
    return out;
}

inline const char* RequireValue(int argc, char** argv, int* index, const char* name)
{
    if (*index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
    }
    ++(*index);
    return argv[*index];
}

inline MoeCombineArgs DefaultArgs()
{
    MoeCombineArgs args;
    args.shape.ep = 2;
    args.shape.m = 64;
    args.shape.k = 7168;
    args.shape.topK = 8;
    args.shape.expertPerRank = 2;
    args.shape.expertNum = args.shape.ep * args.shape.expertPerRank;
    args.shape.maxOutputSize = 0;
    args.shape.aivBlocks = 0;
    args.runtime.deviceBase = 0;
    args.runtime.ndevices = args.shape.ep;
    args.runtime.rankFromMpi = 1;
    args.runtime.rank = 0;
    args.runtime.nranks = args.shape.ep;
    args.runtime.debug = 0;
    args.runtime.iters = 1;
    args.runtime.warmup = 1;
    args.runtime.seed = 1234;
    args.runtime.genData = 1;
    args.runtime.verify = 1;
    args.runtime.skipRun = 0;
    args.runtime.skipBuild = 0;
    args.runtime.cleanBuild = 1;
    args.runtime.skipKernels = 0;
    args.runtime.hostGoldenOnly = 0;
    args.runtime.combineReturnOnly = 0;
    args.runtime.keepHcclShm = 0;
    args.runtime.hcclBuffSizeMb = 0;
    return args;
}

enum class ArgOption {
    kInvalid,
    kCasePreset,
    kRunMode,
    kSocVersion,
    kPes,
    kNranks,
    kTokens,
    kHidden,
    kTopK,
    kExpertsPerRank,
    kMaxOutputSize,
    kAivBlocks,
    kDeviceBase,
    kNdevices,
    kMpiBin,
    kHcclBuffSize,
    kKeepHcclShm,
    kRankFromMpi,
    kRank,
    kDebug,
    kIters,
    kWarmup,
    kSeed,
    kDataDir,
    kGenData,
    kVerify,
    kRtol,
    kAtol,
    kSkipRun,
    kSkipBuild,
    kCleanBuild,
    kSkipKernels,
    kHostGoldenOnly,
    kCombineReturnOnly,
};

struct ArgAlias {
    const char* name;
    ArgOption option;
};

static constexpr ArgAlias kArgAliases[] = {
    {"--case", ArgOption::kCasePreset},
    {"--case-all", ArgOption::kCasePreset},
    {"-r", ArgOption::kRunMode},
    {"--run-mode", ArgOption::kRunMode},
    {"-v", ArgOption::kSocVersion},
    {"--soc-version", ArgOption::kSocVersion},
    {"-pes", ArgOption::kPes},
    {"--pes", ArgOption::kPes},
    {"--nranks", ArgOption::kNranks},
    {"-M", ArgOption::kTokens},
    {"--tokens", ArgOption::kTokens},
    {"-K", ArgOption::kHidden},
    {"--hidden", ArgOption::kHidden},
    {"-topK", ArgOption::kTopK},
    {"--topk", ArgOption::kTopK},
    {"-expertPerPe", ArgOption::kExpertsPerRank},
    {"--experts-per-rank", ArgOption::kExpertsPerRank},
    {"--max-output-size", ArgOption::kMaxOutputSize},
    {"-aivBlocks", ArgOption::kAivBlocks},
    {"--aiv-blocks", ArgOption::kAivBlocks},
    {"-device-base", ArgOption::kDeviceBase},
    {"--device-base", ArgOption::kDeviceBase},
    {"--first-device", ArgOption::kDeviceBase},
    {"--ndevices", ArgOption::kNdevices},
    {"--mpi-bin", ArgOption::kMpiBin},
    {"--hccl-buffsize-mb", ArgOption::kHcclBuffSize},
    {"--keep-hccl-shm", ArgOption::kKeepHcclShm},
    {"--rank-from-mpi", ArgOption::kRankFromMpi},
    {"--rank", ArgOption::kRank},
    {"-debug", ArgOption::kDebug},
    {"--debug", ArgOption::kDebug},
    {"-iters", ArgOption::kIters},
    {"--iters", ArgOption::kIters},
    {"-warmup", ArgOption::kWarmup},
    {"--warmup", ArgOption::kWarmup},
    {"--seed", ArgOption::kSeed},
    {"--data-dir", ArgOption::kDataDir},
    {"--gen-data", ArgOption::kGenData},
    {"--verify", ArgOption::kVerify},
    {"--rtol", ArgOption::kRtol},
    {"--atol", ArgOption::kAtol},
    {"--skip-run", ArgOption::kSkipRun},
    {"--skip-build", ArgOption::kSkipBuild},
    {"--clean-build", ArgOption::kCleanBuild},
    {"--skip-kernels", ArgOption::kSkipKernels},
    {"--host-golden-only", ArgOption::kHostGoldenOnly},
    {"--combine-return-only", ArgOption::kCombineReturnOnly},
};

inline ArgOption FindArgOption(const std::string& key)
{
    for (const ArgAlias& alias : kArgAliases) {
        if (key == alias.name) {
            return alias.option;
        }
    }
    return ArgOption::kInvalid;
}

inline std::string NextArgValue(int argc, char** argv, int* index, const std::string& key)
{
    return std::string(RequireValue(argc, argv, index, key.c_str()));
}

inline bool ParseShapeOption(
    ArgOption option, int argc, char** argv, int* index, const std::string& key, MoeCombineArgs* args)
{
    switch (option) {
        case ArgOption::kPes:
            args->shape.ep = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kNranks:
            args->shape.ep = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            args->runtime.nranks = args->shape.ep;
            args->nranksSet = true;
            return true;
        case ArgOption::kTokens:
            args->shape.m = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kHidden:
            args->shape.k = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kTopK:
            args->shape.topK = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kExpertsPerRank:
            args->shape.expertPerRank = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kMaxOutputSize:
            args->shape.maxOutputSize = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kAivBlocks:
            args->shape.aivBlocks = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        default:
            return false;
    }
}

inline bool ParseRuntimeOption(
    ArgOption option, int argc, char** argv, int* index, const std::string& key, MoeCombineArgs* args,
    bool* ndevicesSet)
{
    switch (option) {
        case ArgOption::kRunMode:
            args->runMode = NextArgValue(argc, argv, index, key);
            return true;
        case ArgOption::kSocVersion:
            args->socVersion = NextArgValue(argc, argv, index, key);
            return true;
        case ArgOption::kDeviceBase:
            args->runtime.deviceBase = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kNdevices:
            args->runtime.ndevices = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            *ndevicesSet = true;
            return true;
        case ArgOption::kMpiBin:
            args->mpiBin = NextArgValue(argc, argv, index, key);
            return true;
        case ArgOption::kHcclBuffSize:
            args->runtime.hcclBuffSizeMb = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kKeepHcclShm:
            args->runtime.keepHcclShm = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kRankFromMpi:
            args->runtime.rankFromMpi = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kRank:
            args->runtime.rank = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            args->rankSet = true;
            return true;
        default:
            return false;
    }
}

inline bool ParseDataOption(
    ArgOption option, int argc, char** argv, int* index, const std::string& key, MoeCombineArgs* args)
{
    switch (option) {
        case ArgOption::kDebug:
            args->runtime.debug = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kIters:
            args->runtime.iters = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kWarmup:
            args->runtime.warmup = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kSeed:
            args->runtime.seed = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kDataDir:
            args->dataDir = NextArgValue(argc, argv, index, key);
            return true;
        case ArgOption::kGenData:
            args->runtime.genData = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kVerify:
            args->runtime.verify = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kRtol:
            args->rtol = ParseDouble(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kAtol:
            args->atol = ParseDouble(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        default:
            return false;
    }
}

inline bool ParseBuildOption(
    ArgOption option, int argc, char** argv, int* index, const std::string& key, MoeCombineArgs* args)
{
    switch (option) {
        case ArgOption::kSkipRun:
            args->runtime.skipRun = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kSkipBuild:
            args->runtime.skipBuild = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kCleanBuild:
            args->runtime.cleanBuild = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kSkipKernels:
            args->runtime.skipKernels = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kHostGoldenOnly:
            args->runtime.hostGoldenOnly = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        case ArgOption::kCombineReturnOnly:
            args->runtime.combineReturnOnly = ParseU32(NextArgValue(argc, argv, index, key), key.c_str());
            return true;
        default:
            return false;
    }
}

inline void FinalizeParsedArgs(MoeCombineArgs* args, bool ndevicesSet)
{
    args->shape.expertNum = args->shape.ep * args->shape.expertPerRank;
    if (args->shape.maxOutputSize == 0) {
        args->shape.maxOutputSize = args->shape.ep * args->shape.m * args->shape.topK;
    }
    if (!ndevicesSet) {
        args->runtime.ndevices = args->shape.ep;
    }
    if (!args->nranksSet) {
        args->runtime.nranks = args->shape.ep;
    }
    if (args->shape.aivBlocks == 0) {
        args->shape.aivBlocks = ChooseDefaultAivBlocks(args->socVersion, args->shape);
    }
}

inline MoeCombineArgs ParseArgs(int argc, char** argv)
{
    MoeCombineArgs args = DefaultArgs();
    bool ndevicesSet = false;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        ArgOption option = FindArgOption(key);
        if (option == ArgOption::kInvalid) {
            throw std::invalid_argument("unknown option: " + key);
        }
        if (option == ArgOption::kCasePreset) {
            throw std::invalid_argument("case presets are unsupported; pass explicit shape parameters");
        }
        if (ParseShapeOption(option, argc, argv, &i, key, &args)) {
            continue;
        }
        if (ParseRuntimeOption(option, argc, argv, &i, key, &args, &ndevicesSet)) {
            continue;
        }
        if (ParseDataOption(option, argc, argv, &i, key, &args)) {
            continue;
        }
        if (!ParseBuildOption(option, argc, argv, &i, key, &args)) {
            throw std::invalid_argument("unhandled option: " + key);
        }
    }
    FinalizeParsedArgs(&args, ndevicesSet);
    return args;
}

inline void ValidateArgs(const MoeCombineArgs& args)
{
    const MoeCombineShape& shape = args.shape;
    const MoeCombineResourceConfig resource = GetResourceConfig(args.socVersion);
    if (args.runMode != "npu") {
        throw std::invalid_argument("run-mode must be npu for the first version");
    }
    if (shape.ep == 0 || shape.m == 0 || shape.k == 0 || shape.topK == 0 || shape.expertPerRank == 0) {
        throw std::invalid_argument("shape fields must be nonzero");
    }
    if (shape.aivBlocks == 0 || shape.aivBlocks > resource.maxAivBlocks) {
        throw std::invalid_argument("aivBlocks must be in [1, maxAivBlocks]");
    }
    uint64_t requiredRows = static_cast<uint64_t>(shape.ep) * shape.m * shape.topK;
    if (shape.maxOutputSize < requiredRows) {
        throw std::invalid_argument("maxOutputSize is smaller than EP * M * topK; capacity/drop is unsupported");
    }
    if (static_cast<uint64_t>(args.runtime.deviceBase) + shape.ep > args.runtime.ndevices) {
        throw std::invalid_argument("deviceBase + pes > ndevices");
    }
}

inline void PrintRunSummary(const MoeCombineArgs& args)
{
    const MoeCombineShape& shape = args.shape;
    std::cout << "RUN_MODE=" << args.runMode << "\n";
    std::cout << "SOC_VERSION=" << args.socVersion << "\n";
    std::cout << "PES=" << shape.ep << " DEVICE_BASE=" << args.runtime.deviceBase
              << " NDEVICES=" << args.runtime.ndevices << "\n";
    std::cout << "M=" << shape.m << " K=" << shape.k << " TOPK=" << shape.topK
              << " EXPERT_PER_PE=" << shape.expertPerRank << " MAX_OUTPUT_SIZE=" << shape.maxOutputSize << "\n";
    std::cout << "AIV_BLOCKS=" << shape.aivBlocks << " TILE_COLS=" << kMoeCombineTileCols
              << " ROW_CHUNK=" << kMoeCombineRowChunk << " METADATA_PAD=" << kMoeCombineMetadataPad << "\n";
    const MoeCombineResourceConfig resource = GetResourceConfig(args.socVersion);
    std::cout << "RESOURCE defaultAicBlocks=" << resource.defaultAicBlocks
              << " defaultAivRatio=" << resource.defaultAivRatio << " defaultAivBlocks=" << resource.defaultAivBlocks
              << " maxAivBlocks=" << resource.maxAivBlocks << "\n";
    std::cout << "DATA_DIR=" << args.dataDir << " SEED=" << args.runtime.seed << "\n";
    std::cout << "WARMUP=" << args.runtime.warmup << " ITERS=" << args.runtime.iters
              << " VERIFY=" << args.runtime.verify << " DEBUG=" << args.runtime.debug << "\n";
}

} // namespace moe_combine

#endif // MOE_COMBINE_ARGS_H_
