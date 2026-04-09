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
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <pto/pto-inst.hpp>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

using namespace pto;

namespace {

template <typename TileData>
void FillLinear(TileData &tile, typename TileData::DType start = typename TileData::DType(1))
{
    auto value = start;
    for (int r = 0; r < tile.GetValidRow(); ++r) {
        for (int c = 0; c < tile.GetValidCol(); ++c) {
            tile.data()[GetTileElementOffset<TileData>(r, c)] = value++;
        }
    }
}

template <typename TileData>
void FillAll(TileData &tile, typename TileData::DType value)
{
    std::fill(tile.data(), tile.data() + TileData::Numel, value);
}

template <typename TileData>
void AssignTileStorage(TileData &tile, size_t &addr)
{
    TASSIGN(tile, addr);
    addr += sizeof(typename TileData::DType) * static_cast<size_t>(TileData::Numel);
    addr = (addr + 63) & ~static_cast<size_t>(63);
}

template <typename... TileData>
void AssignTileStorage(size_t &addr, TileData &...tiles)
{
    (AssignTileStorage(tiles, addr), ...);
}

template <typename TileData>
auto GetValue(const TileData &tile, int r, int c) -> typename TileData::DType
{
    return tile.data()[GetTileElementOffset<TileData>(r, c)];
}

template <typename TileData>
void SetValue(TileData &tile, int r, int c, typename TileData::DType value)
{
    tile.data()[GetTileElementOffset<TileData>(r, c)] = value;
}

template <typename AccTile, typename LeftTile, typename RightTile>
std::vector<typename AccTile::DType> ComputeMatmulExpected(const LeftTile &lhs, const RightTile &rhs,
                                                           const AccTile *acc = nullptr, const float *bias = nullptr)
{
    std::vector<typename AccTile::DType> expected(AccTile::Numel, typename AccTile::DType(0));
    for (int r = 0; r < lhs.GetValidRow(); ++r) {
        for (int c = 0; c < rhs.GetValidCol(); ++c) {
            typename AccTile::DType value = acc ? GetValue(*acc, r, c) : typename AccTile::DType(0);
            for (int k = 0; k < lhs.GetValidCol(); ++k) {
                value += static_cast<typename AccTile::DType>(GetValue(lhs, r, k)) *
                         static_cast<typename AccTile::DType>(GetValue(rhs, k, c));
            }
            if (bias != nullptr) {
                value += static_cast<typename AccTile::DType>(bias[c]);
            }
            expected[GetTileElementOffset<AccTile>(r, c)] = value;
        }
    }
    return expected;
}

template <typename TileData>
void ExpectTileEqualsVector(const TileData &tile, const std::vector<typename TileData::DType> &expected)
{
    ASSERT_EQ(expected.size(), static_cast<size_t>(TileData::Numel));
    for (int i = 0; i < TileData::Numel; ++i) {
        if constexpr (std::is_floating_point_v<typename TileData::DType>) {
            EXPECT_FLOAT_EQ(tile.data()[i], expected[i]);
        } else {
            EXPECT_EQ(tile.data()[i], expected[i]);
        }
    }
}

std::filesystem::path RepoRoot()
{
    auto path = std::filesystem::path(__FILE__).lexically_normal();
    for (int i = 0; i < 6; ++i) {
        path = path.parent_path();
    }
    return path;
}

std::vector<std::string> LoadIsaList(const std::filesystem::path &repoRoot)
{
    std::vector<std::string> ops;
    std::ifstream in(repoRoot / "include/pto/common/pto_instr.hpp");
    std::string line;
    const std::regex recordPattern(R"(^PTO_INST\s+RecordEvent\s+([A-Z0-9_]+)\()");
    while (std::getline(in, line)) {
        std::smatch match;
        if (std::regex_search(line, match, recordPattern)) {
            const std::string name = match[1].str();
            if (std::find(ops.begin(), ops.end(), name) == ops.end()) {
                ops.push_back(name);
            }
        }
    }
    return ops;
}

std::map<std::string, std::set<std::string>> CollectCoverage(const std::filesystem::path &repoRoot,
                                                             const std::vector<std::filesystem::path> &roots,
                                                             const std::vector<std::string> &ops)
{
    std::map<std::string, std::set<std::string>> usage;
    std::set<std::string> isaSet;
    for (const auto &op : ops) {
        usage.emplace(op, std::set<std::string>{});
        isaSet.insert(op);
    }

    const std::regex tokenPattern(R"(\b([A-Z][A-Z0-9_]+)\s*(?:<|\())");

    for (const auto &root : roots) {
        if (!std::filesystem::exists(root)) {
            continue;
        }
        for (const auto &entry : std::filesystem::recursive_directory_iterator(root)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const auto ext = entry.path().extension().string();
            if (ext != ".cpp" && ext != ".hpp" && ext != ".cc" && ext != ".cxx") {
                continue;
            }

            std::ifstream in(entry.path());
            const std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            const std::string label = std::filesystem::relative(entry.path().parent_path(), repoRoot).string();
            for (std::sregex_iterator it(body.begin(), body.end(), tokenPattern), end; it != end; ++it) {
                const std::string op = (*it)[1].str();
                if (isaSet.count(op)) {
                    usage[op].insert(label);
                }
            }
        }
    }

    return usage;
}

std::set<std::string> CollectCpuCaseDirs(const std::filesystem::path &repoRoot)
{
    std::set<std::string> dirs;
    const auto testcaseRoot = repoRoot / "tests/cpu/st/testcase";
    for (const auto &entry : std::filesystem::directory_iterator(testcaseRoot)) {
        if (entry.is_directory()) {
            dirs.insert(entry.path().filename().string());
        }
    }
    return dirs;
}

std::set<std::string> CollectCpuListedCases(const std::filesystem::path &repoRoot)
{
    std::set<std::string> listed;
    std::ifstream in(repoRoot / "tests/cpu/st/testcase/CMakeLists.txt");
    std::string line;
    bool inList = false;
    while (std::getline(in, line)) {
        if (line.find("set(ALL_TESTCASES") != std::string::npos) {
            inList = true;
            continue;
        }
        if (!inList) {
            continue;
        }
        if (line.find(')') != std::string::npos) {
            break;
        }
        const auto begin = line.find_first_not_of(" \t");
        if (begin == std::string::npos || line[begin] == '#') {
            continue;
        }
        const auto end = line.find_last_not_of(" \t");
        listed.insert(line.substr(begin, end - begin + 1));
    }
    return listed;
}

class IsaCoverageTest : public testing::Test {};

TEST_F(IsaCoverageTest, RepoWideCoverageTouchesEveryIsaEntryPoint)
{
    const auto repoRoot = RepoRoot();
    const auto ops = LoadIsaList(repoRoot);
    const auto usage =
        CollectCoverage(repoRoot, {repoRoot / "tests/cpu", repoRoot / "tests/npu", repoRoot / "tests/costmodel"}, ops);

    std::vector<std::string> missing;
    for (const auto &op : ops) {
        if (usage.at(op).empty()) {
            missing.push_back(op);
        }
    }

    EXPECT_TRUE(missing.empty()) << "ISA ST coverage missing for: " << ::testing::PrintToString(missing);
}

TEST_F(IsaCoverageTest, CpuCoverageTouchesEveryIsaEntryPoint)
{
    const auto repoRoot = RepoRoot();
    const auto ops = LoadIsaList(repoRoot);
    const auto usage = CollectCoverage(repoRoot, {repoRoot / "tests/cpu"}, ops);

    std::vector<std::string> missing;
    for (const auto &op : ops) {
        if (usage.at(op).empty()) {
            missing.push_back(op);
        }
    }

    EXPECT_TRUE(missing.empty()) << "ISA ST coverage missing for: " << ::testing::PrintToString(missing);
}

TEST_F(IsaCoverageTest, CpuCaseDirectoriesAreListedInCpuStCMake)
{
    const auto repoRoot = RepoRoot();
    auto dirs = CollectCpuCaseDirs(repoRoot);
    dirs.erase("CMakeLists.txt");
    const auto listed = CollectCpuListedCases(repoRoot);

    std::vector<std::string> missingFromCMake;
    for (const auto &dir : dirs) {
        if (!listed.count(dir)) {
            missingFromCMake.push_back(dir);
        }
    }

    EXPECT_TRUE(missingFromCMake.empty())
        << "CPU testcase directories missing from tests/cpu/st/testcase/CMakeLists.txt: "
        << ::testing::PrintToString(missingFromCMake);
}

TEST_F(IsaCoverageTest, TaxpyAccumulatesScaledSource)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;
    TileData dst;
    TileData src;
    size_t addr = 0;
    AssignTileStorage(addr, dst, src);

    FillLinear(dst, 10.0f);
    FillLinear(src, 1.0f);
    TAXPY(dst, src, 0.5f);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const float originalDst = 10.0f + static_cast<float>(r * dst.GetValidCol() + c);
            const float srcValue = 1.0f + static_cast<float>(r * src.GetValidCol() + c);
            EXPECT_FLOAT_EQ(GetValue(dst, r, c), originalDst + srcValue * 0.5f);
        }
    }
}

TEST_F(IsaCoverageTest, TfmodAndTfmodsUseFloatingPointRemainder)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;
    TileData dstVec;
    TileData src0;
    TileData src1;
    TileData dstScalar;
    size_t addr = 0;
    AssignTileStorage(addr, dstVec, src0, src1, dstScalar);

    FillAll(dstVec, 0.0f);
    FillAll(dstScalar, 0.0f);
    for (int r = 0; r < src0.GetValidRow(); ++r) {
        for (int c = 0; c < src0.GetValidCol(); ++c) {
            SetValue(src0, r, c, 10.0f + static_cast<float>(r + c));
            SetValue(src1, r, c, 3.0f + static_cast<float>((r + c) % 3));
        }
    }

    TFMOD(dstVec, src0, src1);
    TFMODS(dstScalar, src0, 4.0f);

    for (int r = 0; r < src0.GetValidRow(); ++r) {
        for (int c = 0; c < src0.GetValidCol(); ++c) {
            EXPECT_FLOAT_EQ(GetValue(dstVec, r, c), std::fmod(GetValue(src0, r, c), GetValue(src1, r, c)));
            EXPECT_FLOAT_EQ(GetValue(dstScalar, r, c), std::fmod(GetValue(src0, r, c), 4.0f));
        }
    }
}

TEST_F(IsaCoverageTest, TfillpadInplaceAndExpandPadRemainingElements)
{
    using InplaceDst = Tile<TileType::Vec, int16_t, 4, 16, BLayout::RowMajor, 4, 16, SLayout::NoneBox,
                            TileConfig::fractalABSize, PadValue::Max>;
    using SrcTile = Tile<TileType::Vec, int16_t, 4, 16, BLayout::RowMajor, 3, 8>;
    using ExpandDst = Tile<TileType::Vec, int16_t, 5, 16, BLayout::RowMajor, 5, 16, SLayout::NoneBox,
                           TileConfig::fractalABSize, PadValue::Max>;

    InplaceDst inplaceDst;
    ExpandDst expandDst;
    SrcTile src;
    size_t addr = 0;
    AssignTileStorage(addr, inplaceDst, expandDst, src);
    FillAll(inplaceDst, 0);
    FillAll(expandDst, 0);
    FillAll(src, 0);
    FillLinear(src, static_cast<int16_t>(1));

    TFILLPAD_INPLACE(inplaceDst, src);
    TFILLPAD_EXPAND(expandDst, src);

    const int16_t pad = std::numeric_limits<int16_t>::max();
    for (int r = 0; r < inplaceDst.GetValidRow(); ++r) {
        for (int c = 0; c < inplaceDst.GetValidCol(); ++c) {
            const bool copied = r < src.GetValidRow() && c < src.GetValidCol();
            EXPECT_EQ(GetValue(inplaceDst, r, c), copied ? GetValue(src, r, c) : pad);
        }
    }
    for (int r = 0; r < expandDst.GetValidRow(); ++r) {
        for (int c = 0; c < expandDst.GetValidCol(); ++c) {
            const bool copied = r < src.GetValidRow() && c < src.GetValidCol();
            EXPECT_EQ(GetValue(expandDst, r, c), copied ? GetValue(src, r, c) : pad);
        }
    }
}

TEST_F(IsaCoverageTest, TshlsAndTshrsApplyScalarShift)
{
    using TileData = Tile<TileType::Vec, int32_t, 2, 8>;
    TileData left;
    TileData right;
    TileData src;
    size_t addr = 0;
    AssignTileStorage(addr, left, right, src);

    FillLinear(src, 1);
    TSHLS(left, src, 2);
    TSHRS(right, src, 1);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const int32_t value = GetValue(src, r, c);
            EXPECT_EQ(GetValue(left, r, c), value << 2);
            EXPECT_EQ(GetValue(right, r, c), value >> 1);
        }
    }
}

TEST_F(IsaCoverageTest, TsubviewExtractsRequestedWindow)
{
    using SrcTile = Tile<TileType::Vec, float, 4, 8>;
    using DstTile = Tile<TileType::Vec, float, 3, 8, BLayout::RowMajor, 3, 6>;
    SrcTile src;
    DstTile dst;
    size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    FillLinear(src, 1.0f);
    TSUBVIEW(dst, src, 1, 2);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            EXPECT_FLOAT_EQ(GetValue(dst, r, c), GetValue(src, r + 1, c + 2));
        }
    }
}

TEST_F(IsaCoverageTest, TconcatAppendsColumns)
{
    using Src0Tile = Tile<TileType::Vec, int32_t, 2, 8>;
    using Src1Tile = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 2, 4>;
    using DstTile = Tile<TileType::Vec, int32_t, 2, 16, BLayout::RowMajor, 2, 12>;
    Src0Tile src0;
    Src1Tile src1;
    DstTile dst;
    size_t addr = 0;
    AssignTileStorage(addr, src0, src1, dst);

    FillLinear(src0, 1);
    FillLinear(src1, 101);
    FillAll(dst, 0);
    TCONCAT(dst, src0, src1);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < src0.GetValidCol(); ++c) {
            EXPECT_EQ(GetValue(dst, r, c), GetValue(src0, r, c));
        }
        for (int c = 0; c < src1.GetValidCol(); ++c) {
            EXPECT_EQ(GetValue(dst, r, src0.GetValidCol() + c), GetValue(src1, r, c));
        }
    }
}

TEST_F(IsaCoverageTest, TinsertCopiesIntoDestinationAtOffset)
{
    using DstTile = Tile<TileType::Vec, float, 4, 16>;
    using SrcTile = Tile<TileType::Vec, float, 2, 8>;
    using FpTile = Tile<TileType::Vec, float, 1, 8>;
    DstTile plainDst;
    DstTile scalarDst;
    DstTile fpDst;
    SrcTile src;
    FpTile fp;
    size_t addr = 0;
    AssignTileStorage(addr, plainDst, scalarDst, fpDst, src, fp);

    FillAll(plainDst, 0.0f);
    FillAll(scalarDst, 0.0f);
    FillAll(fpDst, 0.0f);
    FillLinear(src, 1.0f);
    FillAll(fp, 2.0f);

    TINSERT(plainDst, src, 1, 4);
    TINSERT<DstTile, SrcTile>(scalarDst, src, static_cast<uint64_t>(7), 1, 4);
    TINSERT_FP<DstTile, SrcTile, FpTile>(fpDst, src, fp, 1, 4);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const float value = GetValue(src, r, c);
            EXPECT_FLOAT_EQ(GetValue(plainDst, r + 1, c + 4), value);
            EXPECT_FLOAT_EQ(GetValue(scalarDst, r + 1, c + 4), value);
            EXPECT_FLOAT_EQ(GetValue(fpDst, r + 1, c + 4), value);
        }
    }
}

TEST_F(IsaCoverageTest, TmovFpCopiesSourceTile)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;
    using FpTile = Tile<TileType::Vec, float, 1, 8>;
    TileData src;
    TileData dst;
    FpTile fp;
    size_t addr = 0;
    AssignTileStorage(addr, src, dst, fp);

    FillLinear(src, 3.0f);
    FillAll(dst, 0.0f);
    FillAll(fp, 1.0f);

    TMOV_FP(dst, src, fp);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            EXPECT_FLOAT_EQ(GetValue(dst, r, c), GetValue(src, r, c));
        }
    }
}

TEST_F(IsaCoverageTest, TextractFpSlicesSourceTile)
{
    using SrcTile = Tile<TileType::Vec, float, 4, 8>;
    using DstTile = Tile<TileType::Vec, float, 3, 8, BLayout::RowMajor, 3, 6>;
    using FpTile = Tile<TileType::Vec, float, 1, 8>;
    SrcTile src;
    DstTile dst;
    FpTile fp;
    size_t addr = 0;
    AssignTileStorage(addr, src, dst, fp);

    FillLinear(src, 1.0f);
    FillAll(fp, 1.0f);

    TEXTRACT_FP(dst, src, fp, 1, 2);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            EXPECT_FLOAT_EQ(GetValue(dst, r, c), GetValue(src, r + 1, c + 2));
        }
    }
}

TEST_F(IsaCoverageTest, TpartmulMultipliesOverlapAndCopiesRemainder)
{
    using DstTile = Tile<TileType::Vec, int32_t, 2, 8>;
    using Src0Tile = Tile<TileType::Vec, int32_t, 2, 8>;
    using Src1Tile = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 1, 4>;
    DstTile dst;
    Src0Tile src0;
    Src1Tile src1;
    size_t addr = 0;
    AssignTileStorage(addr, dst, src0, src1);

    FillLinear(src0, 1);
    FillLinear(src1, 10);
    TPARTMUL(dst, src0, src1);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            if (r < src1.GetValidRow() && c < src1.GetValidCol()) {
                EXPECT_EQ(GetValue(dst, r, c), GetValue(src0, r, c) * GetValue(src1, r, c));
            } else {
                EXPECT_EQ(GetValue(dst, r, c), GetValue(src0, r, c));
            }
        }
    }
}

TEST_F(IsaCoverageTest, TprintWritesReadableMatrix)
{
    using TileData = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 2, 4>;
    TileData src;
    size_t addr = 0;
    AssignTileStorage(addr, src);
    FillLinear(src, 1);

    std::ostringstream captured;
    auto *old = std::cout.rdbuf(captured.rdbuf());
#ifdef _DEBUG
    TPRINT(src);
#else
    TPRINT_IMPL(src);
#endif
    std::cout.rdbuf(old);

    EXPECT_EQ(captured.str(), std::string("TPRINT 2x4\n1 2 3 4\n5 6 7 8\n"));
}

TEST_F(IsaCoverageTest, TgetScaleAddrAliasesSourceStorage)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;
    TileData src;
    TileData dst;
    size_t addr = 0;
    AssignTileStorage(addr, src, dst);
    FillLinear(src, 1.0f);

    TGET_SCALE_ADDR(dst, src);

    ASSERT_EQ(dst.data(), src.data());
    src.data()[3] = 42.0f;
    EXPECT_FLOAT_EQ(dst.data()[3], 42.0f);
}

TEST_F(IsaCoverageTest, TstoreFpStoresTileIntoGlobalTensor)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;
    using FpTile = Tile<TileType::Vec, float, 1, 8>;
    using GlobalData = GlobalTensor<float, Shape<1, 1, 1, 2, 8>, Stride<16, 16, 16, 8, 1>>;

    TileData src;
    FpTile fp;
    size_t addr = 0;
    AssignTileStorage(addr, src, fp);
    std::vector<float> buffer(16, 0.0f);
    GlobalData dst(buffer.data());

    FillLinear(src, 1.0f);
    FillAll(fp, 0.5f);
    TSTORE_FP(dst, src, fp);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            EXPECT_FLOAT_EQ(buffer[r * src.GetValidCol() + c], GetValue(src, r, c));
        }
    }
}

TEST_F(IsaCoverageTest, TfreeDiscardsQueuedTileInCpuSimPipe)
{
    constexpr int FifoDepth = 4;
    constexpr int LocalDepth = 0;
    using TileData = Tile<TileType::Vec, float, 2, 8>;
    using Pipe = TPipe<2, Direction::DIR_C2V, sizeof(float) * TileData::Numel, FifoDepth, LocalDepth>;

    std::vector<float> fifoStorage(TileData::Numel * FifoDepth, 0.0f);
    Pipe::reset_for_cpu_sim();
    Pipe pipe(fifoStorage.data(), 0x0, 0x0);

    TileData first;
    TileData second;
    TileData dst;
    size_t addr = 0;
    AssignTileStorage(addr, first, second, dst);
    FillLinear(first, 1.0f);
    FillLinear(second, 101.0f);
    FillAll(dst, 0.0f);

    TPUSH(first, pipe);
    TFREE(pipe);
    TPUSH(second, pipe);
    TPOP(dst, pipe);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            EXPECT_FLOAT_EQ(GetValue(dst, r, c), GetValue(second, r, c));
        }
    }
}

TEST_F(IsaCoverageTest, TpackCopiesValidValues)
{
    using SrcTile = Tile<TileType::Vec, int32_t, 2, 8>;
    using DstTile = Tile<TileType::Vec, int16_t, 2, 16, BLayout::RowMajor, 2, 8>;
    SrcTile src;
    DstTile dst;
    size_t addr = 0;
    AssignTileStorage(addr, src, dst);
    FillLinear(src, 1);
    FillAll(dst, 0);

    TPACK(dst, src);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            EXPECT_EQ(GetValue(dst, r, c), static_cast<int16_t>(GetValue(src, r, c)));
        }
    }
}

TEST_F(IsaCoverageTest, TrandomWrapperProducesDeterministicOutput)
{
    using TileData = Tile<TileType::Vec, uint32_t, 1, 256>;
    TileData first;
    TileData second;
    size_t addr = 0;
    AssignTileStorage(addr, first, second);
    TRandomKey key = {0x12345678u, 0x9abcdef0u};
    TRandomCounter counter = {0u, 1u, 2u, 3u};
    TRandomKey keyCopy = {key[0], key[1]};
    TRandomCounter counterCopy = {counter[0], counter[1], counter[2], counter[3]};

    TRANDOM(first, key, counter);
    TRANDOM(second, keyCopy, counterCopy);

    for (int c = 0; c < first.GetValidCol(); ++c) {
        EXPECT_EQ(GetValue(first, 0, c), GetValue(second, 0, c));
    }
    EXPECT_NE(GetValue(first, 0, 0), GetValue(first, 0, 1));
}

TEST_F(IsaCoverageTest, TcolprodAndTrowprodReduceProducts)
{
    using SrcTile = Tile<TileType::Vec, float, 3, 8>;
    using ColDst = Tile<TileType::Vec, float, 1, 8>;
    using RowDst = Tile<TileType::Vec, float, 3, 8, BLayout::RowMajor, 3, 1>;
    using TmpTile = Tile<TileType::Vec, float, 3, 8>;
    SrcTile src;
    ColDst colDst;
    RowDst rowDst;
    TmpTile tmp;
    size_t addr = 0;
    AssignTileStorage(addr, src, colDst, rowDst, tmp);

    FillLinear(src, 1.0f);
    TCOLPROD(colDst, src);
    TROWPROD(rowDst, src, tmp);

    for (int c = 0; c < src.GetValidCol(); ++c) {
        float expected = 1.0f;
        for (int r = 0; r < src.GetValidRow(); ++r) {
            expected *= GetValue(src, r, c);
        }
        EXPECT_FLOAT_EQ(GetValue(colDst, 0, c), expected);
    }
    for (int r = 0; r < src.GetValidRow(); ++r) {
        float expected = 1.0f;
        for (int c = 0; c < src.GetValidCol(); ++c) {
            expected *= GetValue(src, r, c);
        }
        EXPECT_FLOAT_EQ(GetValue(rowDst, r, 0), expected);
    }
}

TEST_F(IsaCoverageTest, TrowArgmaxAndTrowArgminReturnIndices)
{
    using SrcTile = Tile<TileType::Vec, float, 3, 8>;
    using DstTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 3, 1>;
    using TmpTile = Tile<TileType::Vec, float, 3, 8>;
    SrcTile src;
    DstTile argmax;
    DstTile argmin;
    TmpTile tmp;
    size_t addr = 0;
    AssignTileStorage(addr, src, argmax, argmin, tmp);

    FillAll(src, 0.0f);
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, static_cast<float>((r + 1) * 10 + c));
        }
    }
    SetValue(src, 0, 5, 100.0f);
    SetValue(src, 1, 2, -7.0f);
    SetValue(src, 2, 7, 99.0f);

    TROWARGMAX(argmax, src, tmp);
    TROWARGMIN(argmin, src, tmp);

    EXPECT_EQ(GetValue(argmax, 0, 0), 5);
    EXPECT_EQ(GetValue(argmax, 1, 0), 7);
    EXPECT_EQ(GetValue(argmax, 2, 0), 7);
    EXPECT_EQ(GetValue(argmin, 0, 0), 0);
    EXPECT_EQ(GetValue(argmin, 1, 0), 2);
    EXPECT_EQ(GetValue(argmin, 2, 0), 0);
}

TEST_F(IsaCoverageTest, TcolArgmaxAndTcolArgminReturnIndices)
{
    using SrcTile = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, 4, 4>;
    using DstTile = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 4>;
    using TmpTile = Tile<TileType::Vec, float, 4, 8>;
    SrcTile src;
    DstTile argmax;
    DstTile argmin;
    TmpTile tmp;
    size_t addr = 0;
    AssignTileStorage(addr, src, argmax, argmin, tmp);

    FillAll(src, 0.0f);
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, static_cast<float>((r + 1) * 10 + c));
        }
    }
    SetValue(src, 3, 0, 100.0f);
    SetValue(src, 1, 1, -5.0f);
    SetValue(src, 2, 2, 77.0f);
    SetValue(src, 0, 3, -9.0f);

    TCOLARGMAX(argmax, src, tmp);
    TCOLARGMIN(argmin, src, tmp);

    EXPECT_EQ(GetValue(argmax, 0, 0), 3);
    EXPECT_EQ(GetValue(argmax, 0, 1), 3);
    EXPECT_EQ(GetValue(argmax, 0, 2), 2);
    EXPECT_EQ(GetValue(argmax, 0, 3), 3);
    EXPECT_EQ(GetValue(argmin, 0, 0), 0);
    EXPECT_EQ(GetValue(argmin, 0, 1), 1);
    EXPECT_EQ(GetValue(argmin, 0, 2), 0);
    EXPECT_EQ(GetValue(argmin, 0, 3), 0);
}

TEST_F(IsaCoverageTest, ThistogramWrapperBuildsCumulativeBins)
{
    using SrcTile = Tile<TileType::Vec, uint16_t, 1, 16, BLayout::RowMajor, 1, 8>;
    using DstTile = Tile<TileType::Vec, uint32_t, 1, 256>;
    using IdxTile = Tile<TileType::Vec, uint8_t, 32, 1, BLayout::ColMajor, 1, 1>;
    SrcTile src;
    DstTile dst;
    IdxTile idx;
    size_t addr = 0;
    AssignTileStorage(addr, src, dst, idx);

    FillAll(src, 0);
    FillAll(dst, 0u);
    idx.data()[0] = 0x12u;
    SetValue(src, 0, 0, static_cast<uint16_t>(0x1201u));
    SetValue(src, 0, 1, static_cast<uint16_t>(0x1202u));
    SetValue(src, 0, 2, static_cast<uint16_t>(0x3410u));
    SetValue(src, 0, 3, static_cast<uint16_t>(0x12ffu));
    SetValue(src, 0, 4, static_cast<uint16_t>(0x2211u));
    SetValue(src, 0, 5, static_cast<uint16_t>(0x2212u));
    SetValue(src, 0, 6, static_cast<uint16_t>(0x4413u));
    SetValue(src, 0, 7, static_cast<uint16_t>(0x2214u));

    THISTOGRAM<HistByte::BYTE_1>(dst, src, idx);
    EXPECT_EQ(GetValue(dst, 0, 0x11), 0u);
    EXPECT_EQ(GetValue(dst, 0, 0x12), 3u);
    EXPECT_EQ(GetValue(dst, 0, 0x33), 6u);
    EXPECT_EQ(GetValue(dst, 0, 0x34), 7u);

    THISTOGRAM<HistByte::BYTE_0>(dst, src, idx);
    EXPECT_EQ(GetValue(dst, 0, 0x00), 0u);
    EXPECT_EQ(GetValue(dst, 0, 0x01), 1u);
    EXPECT_EQ(GetValue(dst, 0, 0x02), 2u);
    EXPECT_EQ(GetValue(dst, 0, 0xFE), 2u);
    EXPECT_EQ(GetValue(dst, 0, 0xFF), 3u);
}

TEST_F(IsaCoverageTest, TdequantAppliesScaleAndOffset)
{
    using DstTile = Tile<TileType::Vec, float, 2, 8>;
    using SrcTile = Tile<TileType::Vec, int32_t, 2, 8>;
    using ParaTile = Tile<TileType::Vec, float, 2, 8>;
    DstTile dst;
    SrcTile src;
    ParaTile scale;
    ParaTile offset;
    size_t addr = 0;
    AssignTileStorage(addr, dst, src, scale, offset);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, (r + 1) * 10 + c);
            SetValue(scale, r, c, 0.5f + static_cast<float>(c) * 0.25f);
            SetValue(offset, r, c, static_cast<float>(r + c));
        }
    }

    TDEQUANT(dst, src, scale, offset);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const float expected =
                (static_cast<float>(GetValue(src, r, c)) - GetValue(offset, r, c)) * GetValue(scale, r, c);
            EXPECT_FLOAT_EQ(GetValue(dst, r, c), expected);
        }
    }
}

TEST_F(IsaCoverageTest, TquantScalarAndMxWrappersAreCallable)
{
    using SrcTile = Tile<TileType::Vec, float, 16, 64>;
    using SymDstTile = Tile<TileType::Vec, int8_t, 16, 64>;
    using ParaTile = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
    using Fp8Tile = Tile<TileType::Vec, int8_t, 16, 64>;
    using ExpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, 1, 32>;
    using MaxTile = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32>;
    using IdxTile = Tile<TileType::Vec, uint16_t, 1, 16, BLayout::RowMajor, 1, 16>;

    SrcTile src;
    SrcTile scaling;
    SymDstTile symDst;
    Fp8Tile fp8Dst;
    ParaTile invScale;
    ExpTile exp;
    ExpTile expZz;
    MaxTile max;
    IdxTile gatherIdx;
    size_t addr = 0;
    AssignTileStorage(addr, src, scaling, symDst, fp8Dst, invScale, exp, expZz, max, gatherIdx);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        invScale.data()[GetTileElementOffset<ParaTile>(r, 0)] = 2.0f;
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const float value = static_cast<float>((r + c) % 11) * 0.25f - 1.0f;
            src.data()[GetTileElementOffset<SrcTile>(r, c)] = value;
            scaling.data()[GetTileElementOffset<SrcTile>(r, c)] = 0.0f;
        }
    }
    for (int i = 0; i < gatherIdx.GetValidCol(); ++i) {
        gatherIdx.data()[i] = static_cast<uint16_t>(i);
    }

    TQUANT<QuantType::INT8_SYM>(symDst, src, invScale);
    TQUANT<QuantType::MXFP8>(fp8Dst, src, &exp, &max, &scaling);
    TQUANT<QuantType::MXFP8, VecStoreMode::NZ>(fp8Dst, src, &exp, &max, &scaling, &expZz, &gatherIdx);

    EXPECT_NE(symDst.data()[0], 0);
    EXPECT_NE(static_cast<uint8_t>(fp8Dst.data()[0]), 0u);
    EXPECT_NE(exp.data()[0], 0u);
    EXPECT_NE(expZz.data()[0], 0u);
}

TEST_F(IsaCoverageTest, TgemvAndMxVariantsMatchCpuMatmulSemantics)
{
    using LeftTile = TileLeft<float, 16, 16>;
    using RightTile = TileRight<float, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using BiasTile = Tile<TileType::Bias, float, 1, 16>;
    using LeftScaleTile = TileLeftScale<float, 16, 2>;
    using RightScaleTile = TileRightScale<float, 16, 2>;

    LeftTile lhs;
    RightTile rhs;
    AccTile gemv;
    AccTile gemvAcc;
    AccTile gemvMx;
    AccTile matmulMx;
    AccTile accIn;
    BiasTile bias;
    LeftScaleTile lhsScale;
    RightScaleTile rhsScale;
    size_t addr = 0;
    AssignTileStorage(addr, lhs, rhs, gemv, gemvAcc, gemvMx, matmulMx, accIn, bias, lhsScale, rhsScale);

    FillAll(lhs, 0.0f);
    FillAll(rhs, 0.0f);
    FillAll(accIn, 1.0f);
    FillAll(lhsScale, 2.0f);
    FillAll(rhsScale, 3.0f);
    for (int r = 0; r < lhs.GetValidRow(); ++r) {
        for (int c = 0; c < lhs.GetValidCol(); ++c) {
            SetValue(lhs, r, c, static_cast<float>(r + c + 1));
            SetValue(rhs, r, c, static_cast<float>((r == c) ? 2 : 1));
        }
    }
    for (int c = 0; c < bias.GetValidCol(); ++c) {
        SetValue(bias, 0, c, static_cast<float>(c));
    }

    TGEMV(gemv, lhs, rhs);
    TGEMV_ACC(gemvAcc, accIn, lhs, rhs);
    TGEMV_MX(gemvMx, lhs, lhsScale, rhs, rhsScale);
    TMATMUL_MX(matmulMx, lhs, lhsScale, rhs, rhsScale);

    const auto expectedGemv = ComputeMatmulExpected<AccTile>(lhs, rhs);
    const auto expectedGemvAcc = ComputeMatmulExpected<AccTile>(lhs, rhs, &accIn);
    ExpectTileEqualsVector(gemv, expectedGemv);
    ExpectTileEqualsVector(gemvAcc, expectedGemvAcc);
    ExpectTileEqualsVector(gemvMx, expectedGemv);
    ExpectTileEqualsVector(matmulMx, expectedGemv);

    AccTile gemvBias;
    AssignTileStorage(addr, gemvBias);
    TGEMV_BIAS(gemvBias, lhs, rhs, bias);
    std::vector<float> biasValues(bias.GetValidCol());
    for (int c = 0; c < bias.GetValidCol(); ++c) {
        biasValues[c] = GetValue(bias, 0, c);
    }
    const auto expectedGemvBias = ComputeMatmulExpected<AccTile>(lhs, rhs, nullptr, biasValues.data());
    ExpectTileEqualsVector(gemvBias, expectedGemvBias);
}

} // namespace
