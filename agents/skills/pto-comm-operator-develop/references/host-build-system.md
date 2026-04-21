# Host 侧与构建系统

## 标准初始化流程

```cpp
int main(int argc, char **argv) {
    // 1. MPI 初始化
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // 2. ACL 初始化
    aclInit(nullptr);
    aclrtSetDevice(rank % device_count);
    aclrtStream computeStream, commStream;
    aclrtCreateStream(&computeStream);
    aclrtCreateStream(&commStream);

    // 3. HCCL 通信域创建
    HcclRootInfo rootInfo;
    if (rank == 0) HcclGetRootInfo(&rootInfo);
    MPI_Bcast(&rootInfo, sizeof(rootInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    HcclComm hcclComm;
    HcclCommInitRootInfo(nranks, &rootInfo, rank, &hcclComm);

    // 4. 获取通信上下文（远端地址）

    // 5. 内存分配
    uint8_t *buffer;
    aclrtMalloc((void**)&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST);

    // 6. 信号矩阵初始化（清零）
    aclrtMemset(signal_matrix, 0, signal_size);

    // 7. 启动 kernel
    launchCommKernel(buffer, ..., commStream);
    aclrtSynchronizeStream(commStream);

    // 8. 验证结果

    // 9. 清理
    HcclCommDestroy(hcclComm);
    aclrtDestroyStream(computeStream);
    aclrtDestroyStream(commStream);
    aclrtResetDevice(rank % device_count);
    aclFinalize();
    MPI_Finalize();
}
```

---

## 信号矩阵清零

**关键**：每次 kernel 执行前必须清零信号矩阵，否则上次的残留值导致同步错误。

```cpp
aclrtMemset(signal_matrix, signal_size, 0, signal_size);
aclrtSynchronizeStream(stream);
```

---

## Kernel 启动函数模式

```cpp
// kernel_launchers.h 声明
void launchCommKernel(uint8_t *data, uint8_t *signal, uint8_t *ctx,
                      int rank, int nranks, void *stream);

// comm_kernel.cpp 实现
void launchCommKernel(uint8_t *data, uint8_t *signal, uint8_t *ctx,
                      int rank, int nranks, void *stream)
{
    CommKernelEntry<<<COMM_BLOCK_NUM, nullptr, stream>>>(
        data, signal, ctx, rank, nranks, COMM_BLOCK_NUM);
}
```

---

## CMakeLists.txt 模板

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_comm_operator)

set(CMAKE_CXX_COMPILER bisheng)
set(CMAKE_CXX_STANDARD 17)

# PTO 头文件路径（优先使用仓库内版本）
set(PTO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
include_directories(BEFORE ${PTO_INCLUDE_DIR})

# CANN 环境
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()
include_directories(${ASCEND_HOME}/include)
link_directories(${ASCEND_HOME}/lib64)

# 通信 Kernel（Vec 架构）
add_library(comm_kernel SHARED comm_kernel.cpp)
target_compile_options(comm_kernel PRIVATE
    --cce-aicore-arch=dav-c220-vec
    -DMEMORY_BASE
    -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_options(comm_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(comm_kernel runtime)

# 计算 Kernel（Cube 架构，如需通算融合）
add_library(compute_kernel SHARED compute_kernel.cpp)
target_compile_options(compute_kernel PRIVATE
    --cce-aicore-arch=dav-c220-cube
    -DMEMORY_BASE
    -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_options(compute_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(compute_kernel runtime)

# Host 可执行文件
add_executable(my_operator main.cpp)
target_link_libraries(my_operator
    comm_kernel compute_kernel
    ascendcl hccl tiling_api platform)
```

### 关键配置项

| 配置 | 说明 |
|------|------|
| `--cce-aicore-arch=dav-c220-vec` | 通信 kernel 使用 Vec 架构 |
| `--cce-aicore-arch=dav-c220-cube` | 计算 kernel 使用 Cube 架构 |
| `-DMEMORY_BASE` | 启用远端地址计算宏 |
| `--cce-fatobj-link` | 启用 fat object 链接 |

---

## MPI 运行

```bash
# 单机多卡
mpirun -np 8 ./my_operator

# 多机
mpirun -np 16 -H host1:8,host2:8 ./my_operator
```
