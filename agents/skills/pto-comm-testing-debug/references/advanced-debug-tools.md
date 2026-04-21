# 高级调试工具

## mssanitizer 内存检测

`mssanitizer` 是昇腾平台的内存安全检测工具，可检测通信算子中的内存越界、未对齐访问等问题。

### 编译集成

在 CMakeLists.txt 中添加 sanitizer 编译选项：

```cmake
option(ENABLE_SANITIZER "Enable mssanitizer for memory checking" OFF)

if(ENABLE_SANITIZER)
    target_compile_options(comm_kernel PRIVATE -fsanitize=memory)
    target_link_options(comm_kernel PRIVATE -fsanitize=memory)
    target_compile_options(compute_kernel PRIVATE -fsanitize=memory)
    target_link_options(compute_kernel PRIVATE -fsanitize=memory)
endif()
```

### 使用方式

```bash
# 编译带 sanitizer 的版本
cmake .. -DENABLE_SANITIZER=ON
make -j

# 运行（mssanitizer 自动检测）
mssanitizer --tool=memcheck mpirun -np 8 ./my_operator
```

### 可检测问题

| 问题类型 | 说明 | 通信算子中的典型场景 |
|---------|------|---------------------|
| GM 越界读 | 读取超出分配范围的 GM 地址 | Tile 分块时边界计算错误 |
| GM 越界写 | 写入超出分配范围的 GM 地址 | 远端地址偏移计算溢出 |
| UB 越界 | 访问超出 UB 容量 | Tile 大小设置过大 |
| 未对齐访问 | 未满足对齐要求 | Signal 地址非 4B 对齐 |

### 输出解读

```
[mssanitizer] ERROR: out-of-bounds access at GM address 0x12345678
  in kernel CommKernelEntry at comm_kernel.cpp:142
  allocated at main.cpp:89 with size 65536
  accessed offset: 65540 (4 bytes beyond allocation)
```

重点关注：
- `out-of-bounds`：检查 Tile 边界和远端地址计算
- `use-after-free`：检查 buffer 生命周期
- `uninitialized`：检查信号矩阵是否清零

---

## 环境变量调试

```bash
# HCCL 调试
export HCCL_LOG_LEVEL=DEBUG      # HCCL 日志级别
export HCCL_BUFFSIZE=1024        # 通信缓冲区大小（MB）

# ACL 错误码检查
export ACL_ERROR_ABORT=1         # 遇到 ACL 错误立即 abort
```

---

## 缩小问题规模

```cpp
#define DEBUG_MODE
#ifdef DEBUG_MODE
static constexpr uint32_t G_ORIG_M = 128;
static constexpr uint32_t G_ORIG_N = 256;
static constexpr int COMPUTE_BLOCK_NUM = 2;
static constexpr int COMM_BLOCK_NUM = 2;
#endif
```

---

## Host 侧性能计时

```cpp
aclrtEvent startEvent, endEvent;
aclrtCreateEvent(&startEvent);
aclrtCreateEvent(&endEvent);

aclrtRecordEvent(startEvent, stream);
launchKernel(..., stream);
aclrtRecordEvent(endEvent, stream);
aclrtSynchronizeStream(stream);

float elapsed_ms;
aclrtEventElapsedTime(&elapsed_ms, startEvent, endEvent);
printf("Kernel time: %.3f ms\n", elapsed_ms);
```

---

## Warmup + 多次测量

```cpp
// Warmup（排除首次开销）
for (int i = 0; i < WARMUP_ITERS; i++) {
    ClearSignals();
    LaunchKernel(...);
    aclrtSynchronizeStream(stream);
}

// 正式测量
float total_ms = 0;
for (int i = 0; i < MEASURE_ITERS; i++) {
    ClearSignals();
    aclrtRecordEvent(start, stream);
    LaunchKernel(...);
    aclrtRecordEvent(end, stream);
    aclrtSynchronizeStream(stream);
    float ms;
    aclrtEventElapsedTime(&ms, start, end);
    total_ms += ms;
}
printf("Average: %.3f ms\n", total_ms / MEASURE_ITERS);
```

---

## msprof 硬件 Profiling

对于 Device 侧管道级分析：

```bash
# 采集 kernel 执行 timeline
msprof --output=./prof_data --application="mpirun -np 8 ./my_operator"

# 导出分析结果
msprof --export=timeline --output=./prof_data
```

可展示 MTE2/MTE3/Cube/Vec 管道占用率，定位通信/计算重叠空洞。
