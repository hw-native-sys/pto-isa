# 正确性验证方法

## Golden 数据生成

### Python Golden 生成脚本模板

```python
#!/usr/bin/python3
import os
import numpy as np
np.random.seed(42)

def gen_reduce_scatter_golden(nranks, M, N, dtype=np.float16):
    """生成 ReduceScatter 的 golden 数据"""
    inputs = []
    for r in range(nranks):
        data = np.random.randn(M, N).astype(dtype)
        data.tofile(f"rank{r}_input.bin")
        inputs.append(data)

    summed = np.sum(inputs, axis=0)

    for r in range(nranks):
        golden = np.zeros_like(summed)
        # 按 tiling 策略填充 rank r 应该持有的结果
        golden.tofile(f"rank{r}_golden.bin")

def gen_allgather_golden(nranks, M, N, dtype=np.float16):
    """生成 AllGather 的 golden 数据：每个 rank 持有完整的 summed 结果"""
    inputs = []
    for r in range(nranks):
        data = np.fromfile(f"rank{r}_input.bin", dtype=dtype).reshape(M, N)
        inputs.append(data)

    golden = np.sum(inputs, axis=0)
    for r in range(nranks):
        golden.tofile(f"rank{r}_allreduce_golden.bin")

if __name__ == "__main__":
    nranks = 8
    M, N = 5416, 1408
    gen_reduce_scatter_golden(nranks, M, N)
    gen_allgather_golden(nranks, M, N)
```

### Golden 数据组织

```
testdata/
├── rank0_input.bin
├── rank1_input.bin
├── ...
├── rank0_golden.bin
├── rank1_golden.bin
├── ...
└── config.json
```

### 数据类型映射

| Python numpy | C++ 类型 | ACL 类型 |
|-------------|---------|----------|
| `np.float16` | `half` | `aclFloat16` |
| `np.float32` | `float` | `float` |
| `np.int32` | `int32_t` | `int32_t` |
| `np.int16` | `int16_t` | `int16_t` |

---

## 验证函数模板

```cpp
template <typename T>
bool VerifyResult(const T *actual, const T *expected, size_t count,
                  float atol = 1.0f, float rtol = 0.01f)
{
    int error_count = 0;
    int max_errors = 10;
    float max_diff = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        float a = static_cast<float>(actual[i]);
        float e = static_cast<float>(expected[i]);
        float diff = std::abs(a - e);
        float threshold = atol + rtol * std::abs(e);

        if (diff > threshold) {
            if (error_count < max_errors) {
                printf("  Mismatch at [%zu]: actual=%f, expected=%f, diff=%f, threshold=%f\n",
                       i, a, e, diff, threshold);
            }
            error_count++;
            max_diff = std::max(max_diff, diff);
        }
    }

    if (error_count > 0) {
        printf("  Total errors: %d / %zu (max_diff=%f)\n", error_count, count, max_diff);
    }
    return error_count == 0;
}
```

---

## 精度标准

| 数据类型 | 推荐 atol | 推荐 rtol | 说明 |
|---------|----------|----------|------|
| float (FP32) | 1e-5 | 1e-4 | 高精度 |
| half (FP16) | 1.0 | 0.01 | AtomicAdd 累积误差较大 |
| int32 / int16 | 0 | 0 | 精确匹配 |

**FP16 AtomicAdd 精度注意**：
- 多 rank AtomicAdd 累积会引入浮点误差
- rank 数越多，累积误差越大
- 建议 FP16 使用 `atol=1.0, rtol=0.01` 或更宽松的阈值

---

## 分阶段验证

对于多阶段算子（如 RS + Barrier + AG），建议分阶段验证：

```cpp
// 阶段 1 验证：RS 完成后，检查 reduced_output
RunReduceScatterOnly(...);
aclrtSynchronizeStream(stream);
bool rs_pass = VerifyReduceScatter(reduced_output, rs_golden);

// 阶段 2 验证：完整 AllReduce 后，检查最终结果
RunFullAllReduce(...);
aclrtSynchronizeStream(stream);
bool ar_pass = VerifyAllReduce(reduced_output, ar_golden);
```

---

## main.cpp 模板（多 rank 测试）

```cpp
#include "acl/acl.h"
#include "comm_mpi.h"
#include "hccl_context.h"
#include <cstdio>

extern void launchTPutTest(uint8_t *local, uint8_t *remote, void *stream);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    aclInit(nullptr);
    aclrtSetDevice(rank);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    HcclRootInfo rootInfo;
    if (rank == 0) HcclGetRootInfo(&rootInfo);
    MPI_Bcast(&rootInfo, sizeof(rootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    HcclComm comm;
    HcclCommInitRootInfo(nranks, &rootInfo, rank, &comm);

    size_t dataSize = ROWS * COLS * sizeof(half);
    uint8_t *localBuf, *remoteBuf;
    // ... 获取通信窗口地址 ...

    std::vector<half> hostData(ROWS * COLS);
    for (int i = 0; i < ROWS * COLS; i++) hostData[i] = (half)(rank * 1000 + i);
    aclrtMemcpy(localBuf, dataSize, hostData.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

    MPI_Barrier(MPI_COMM_WORLD);

    launchTPutTest(localBuf, remoteBuf, stream);
    aclrtSynchronizeStream(stream);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<half> result(ROWS * COLS);
    aclrtMemcpy(result.data(), dataSize, localBuf, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    bool pass = true;
    for (int i = 0; i < ROWS * COLS; i++) {
        half expected = /* 根据通信语义计算 */;
        if (abs((float)result[i] - (float)expected) > 1e-3) {
            printf("FAIL: rank %d, idx %d, got %f, expected %f\n",
                   rank, i, (float)result[i], (float)expected);
            pass = false;
            break;
        }
    }
    printf("Rank %d: %s\n", rank, pass ? "PASS" : "FAIL");

    HcclCommDestroy(comm);
    aclrtDestroyStream(stream);
    aclrtResetDevice(rank);
    aclFinalize();
    MPI_Finalize();
    return pass ? 0 : 1;
}
```
