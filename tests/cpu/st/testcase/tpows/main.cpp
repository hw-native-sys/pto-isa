#include "test_common.h"

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTPOWSTestCase(void *out, void *src, void *scalar, aclrtStream stream);

class TPOWSTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int oRow, int oCol>
inline void InitDstDevice(T *dstDevice)
{
    constexpr int size = oRow * oCol;
    for (int i = 0; i < size; ++i) {
        dstDevice[i] = T{0};
    }
}

template <uint32_t caseId, typename T, int validRow, int validCol, int iRow = validRow, int iCol = validCol,
          int oRow = validRow, int oCol = validCol>
bool TPowSTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = oRow * oCol * sizeof(T);
    size_t srcByteSize = iRow * iCol * sizeof(T);
    size_t readSize = 0;
    T *dstHost;
    T *srcHost;
    T *dstDevice;
    T *srcDevice;
    T scalar;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);

    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    InitDstDevice<T, oRow, oCol>(dstDevice);
    ReadFile(GetGoldenDir() + "/input.bin", readSize, srcHost, srcByteSize);

    std::ifstream file(GetGoldenDir() + "/scalar.bin", std::ios::binary);

    file.read(reinterpret_cast<char *>(&scalar), sizeof(T));
    file.close();
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTPOWSTestCase<caseId>(dstDevice, srcDevice, &scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(oRow * oCol);
    std::vector<T> devFinal(oRow * oCol);
    size_t goldenReadSize = 0;
    size_t outputReadSize = 0;
    ReadFile(GetGoldenDir() + "/golden.bin", goldenReadSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputReadSize, devFinal.data(), dstByteSize);

    return ResultCmp<T>(golden, devFinal, 0.001f);
}

TEST_F(TPOWSTest, case1)
{
    bool ret = TPowSTestFramework<1, float, 32, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TPOWSTest, case2)
{
    bool ret = TPowSTestFramework<2, aclFloat16, 63, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TPOWSTest, case3)
{
    bool ret = TPowSTestFramework<3, int32_t, 31, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TPOWSTest, case4)
{
    bool ret = TPowSTestFramework<4, int16_t, 15, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TPOWSTest, case5)
{
    bool ret = TPowSTestFramework<5, float, 7, 448>();
    EXPECT_TRUE(ret);
}

TEST_F(TPOWSTest, case6)
{
    bool ret = TPowSTestFramework<6, float, 256, 16>();
    EXPECT_TRUE(ret);
}

TEST_F(TPOWSTest, case7)
{
    bool ret = TPowSTestFramework<7, float, 16, 16, 32, 32, 64, 64>();
    EXPECT_TRUE(ret);
}
