# 测试环境与运行

## CPU 仿真测试

### 适用范围

CPU 仿真支持以下同步通信指令的功能验证：
- TPUT、TGET（P2P 传输）
- TNOTIFY、TWAIT、TTEST（信号同步）
- TGATHER、TSCATTER、TBROADCAST、TREDUCE（集合通信）

**不支持**：异步指令（TPUT_ASYNC/TGET_ASYNC）在 CPU 仿真下直接返回空 event。

### 运行方式

```bash
python3 tests/run_cpu.py --testcase <testcase_name> --gtest_filter '<filter>'

# 示例
python3 tests/run_cpu.py --testcase tgather --gtest_filter 'TGatherTest.*'
```

### CPU 仿真局限性

| 特性 | CPU 仿真 | NPU 硬件 |
|------|---------|---------|
| 功能正确性 | 验证数据计算逻辑 | 验证完整执行 |
| 远端地址 | 模拟为本地指针 | 真实硬件远端 DMA |
| 信号同步 | 模拟 AtomicAdd/Set | 硬件原子操作 |
| pipe_barrier | 忽略 | 真实流水线同步 |
| 多 rank | 单进程模拟 | MPI + 多 NPU |
| 异步 DMA | 返回无效 event | SDMA/URMA 引擎 |

**建议**：CPU 仿真验证数据流和逻辑正确性，但最终必须在 NPU 硬件上验证同步和性能。

---

## NPU 硬件测试

### 单指令 ST 测试

```bash
python3 tests/script/run_st.py -r npu -v a3 -t tput -g TPutTest.*

# 运行所有通信 ST
python3 tests/script/run_st.py -r npu -v a3 --comm
```

### 算子级测试

```bash
cd kernels/manual/a2a3/my_operator
mkdir -p build && cd build
cmake .. -DSOC_VERSION=Ascend910C -DRUN_MODE=npu
make -j

# 运行（8 rank）
mpirun -np 8 ./my_operator
```

### 测试 Kernel 结构

```cpp
template <typename T, int Rows, int Cols>
__global__ AICORE void TPutTestKernel(__gm__ T *local_data, __gm__ T *remote_addr)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using Global = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using TileT = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;

    ShapeDyn shape(1, 1, 1, Rows, Cols);
    StrideDyn stride(Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1);

    Global srcG(local_data, shape, stride);
    Global dstG(remote_addr, shape, stride);

    TileT stagingTile(Rows, Cols);
    TASSIGN(stagingTile, 0x0);

    comm::TPUT(dstG, srcG, stagingTile);
}
```

---

## 多 Rank 测试框架

### MPI Wrapper

使用动态加载避免硬依赖：

```cpp
class MpiWrapper {
    void *handle_;
    int (*MPI_Init_)(int*, char***);
    int (*MPI_Comm_rank_)(MPI_Comm, int*);
public:
    MpiWrapper() {
        handle_ = dlopen("libmpi.so", RTLD_LAZY);
        MPI_Init_ = (decltype(MPI_Init_))dlsym(handle_, "MPI_Init");
    }
};
```

### 测试运行脚本模板

```bash
#!/bin/bash
set -e

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

mkdir -p build && cd build
cmake .. -DSOC_VERSION=${SOC_VERSION:-Ascend910C} -DRUN_MODE=npu
make -j$(nproc)
cd ..

NRANKS=${NRANKS:-8}
export HCCL_BUFFSIZE=1024

mpirun -np $NRANKS \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH \
    -x ASCEND_HOME_PATH \
    ./build/my_operator "$@"
```

### Rank 0 收集结果

```cpp
if (rank == 0) {
    bool pass = VerifyResult(actual, golden, count);
    int result = pass ? 0 : 1;
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Overall: %s\n", pass ? "PASS" : "FAIL");
} else {
    int result;
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
}
```
