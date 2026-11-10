# PTO 通信 ISA 参考手册

本目录包含 PTO 通信 ISA 的逐指令参考文档。

- 权威来源（C++ 内建接口）：`include/pto/comm/pto_comm_inst.hpp`
- 类型定义：`include/pto/comm/comm_types.hpp`

## 点对点通信（同步）
- [**TPUT**](TPUT_zh.md)：远程写（GM → UB → GM）
- [**TGET**](TGET_zh.md)：远程读（GM → UB → GM）

## 点对点通信（异步）
- [**TPUT_ASYNC**](TPUT_ASYNC_zh.md)：异步远程写（GM → DMA 引擎 → GM）
- [**TPUT_ASYNC_NOTIFY**](TPUT_ASYNC_NOTIFY_zh.md)：远程写并更新远端 `int32_t` signal
- [**TGET_ASYNC**](TGET_ASYNC_zh.md)：异步远程读（GM → DMA 引擎 → GM）

## 基于信号的同步
- [**TNOTIFY**](TNOTIFY_zh.md)：向远端 NPU 发送通知
- [**TWAIT**](TWAIT_zh.md)：阻塞等待信号条件满足
- [**TTEST**](TTEST_zh.md)：非阻塞检测信号条件

## 集合通信

- [**TGATHER**](TGATHER_zh.md)：从所有 rank 收集数据
- [**TSCATTER**](TSCATTER_zh.md)：向所有 rank 分发数据
- [**TREDUCE**](TREDUCE_zh.md)：从所有 rank 归约数据到本地
- [**TBROADCAST**](TBROADCAST_zh.md)：从当前 NPU 广播数据到所有 rank

## 类型定义

### NotifyOp

`TNOTIFY` 的操作类型：

| 值 | 说明 |
|-------|-------------|
| `NotifyOp::Set` | 直接赋值（`signal = value`）|
| `NotifyOp::AtomicAdd` | 原子加（`signal += value`）|

### WaitCmp

`TWAIT` 和 `TTEST` 的比较运算符：

| 值 | 说明 |
|-------|-------------|
| `WaitCmp::EQ` | 等于（`==`）|
| `WaitCmp::NE` | 不等于（`!=`）|
| `WaitCmp::GT` | 大于（`>`）|
| `WaitCmp::GE` | 大于等于（`>=`）|
| `WaitCmp::LT` | 小于（`<`）|
| `WaitCmp::LE` | 小于等于（`<=`）|

```cpp
// 用法示例（统一运行时参数风格）：
comm::TNOTIFY(signal, 1, comm::NotifyOp::Set);
comm::TWAIT(signal, 1, comm::WaitCmp::EQ);
comm::TTEST(signal, 1, comm::WaitCmp::GE);
```

### ReduceOp

`TREDUCE` 的归约运算符：

| 值 | 说明 |
|-------|-------------|
| `ReduceOp::Sum` | 逐元素求和 |
| `ReduceOp::Max` | 逐元素取最大值 |
| `ReduceOp::Min` | 逐元素取最小值 |

### AtomicType

`TPUT` 的原子操作类型（定义于 `include/pto/common/constants.hpp`）：

| 值 | 说明 |
|-------|-------------|
| `AtomicType::AtomicNone` | 无原子操作（默认）|
| `AtomicType::AtomicAdd` | 原子加操作 |

### DmaEngine

`TPUT_ASYNC`、`TPUT_ASYNC_NOTIFY` 和 `TGET_ASYNC` 的 DMA 后端选择：

| 值 | 说明 |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA 引擎（支持一维传输，Ascend950 上仅支持TGET|
| `DmaEngine::URMA` | URMA 引擎（支持一维传输，仅 Ascend950 / NPU_ARCH 3510；要求 CANN >= 9.1.0）|
| `DmaEngine::RDMA` | RDMA 引擎（支持一维传输，仅 Ascend950 / NPU_ARCH 3510；当前网卡平台仅支持 HNS1825）|

### AsyncSubmitMode

`AsyncSession::submitMode` 的取值，控制 `TPUT_ASYNC` 立即提交还是暂存后成批提交：

| 值 | 说明 |
|-------|-------------|
| `AsyncSubmitMode::IMMEDIATE` | 立即提交（默认）。若存在暂存批次，先提交该批次 |
| `AsyncSubmitMode::DEFER` | 暂存本次操作，暂存数达到 `batchSize` 时自动提交 |
| `AsyncSubmitMode::DEFER_AND_SUBMIT` | 暂存本次操作并立即提交整个批次 |

仅 A2/A3 的 `TPUT_ASYNC<SDMA>` 与 Ascend 950 的 `TPUT_ASYNC<URMA>` 支持批量。详见 [TPUT_ASYNC 批量提交](TPUT_ASYNC_zh.md#批量提交)。

### AsyncEvent

由 `TPUT_ASYNC` / `TPUT_ASYNC_NOTIFY` / `TGET_ASYNC` 返回，用于同步传输完成状态：

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                           // 仅表示 handle != 0
    bool Wait(const AsyncSession &session) const; // 阻塞直到传输完成
    bool Test(const AsyncSession &session) const; // 非阻塞完成检测

    // 供无法执行设备侧轮询原语的消费者读取完成记录
    uint32_t CompletionRecordCount(const AsyncSession &session) const;
    AsyncCompletionRecord CompletionRecordAt(const AsyncSession &session, uint32_t idx) const;
};
```

`Wait` 和 `Test` 的返回值是判断操作结果的唯一依据：返回 `true` 表示操作已完成，返回 `false` 表示提交失败。`handle == 0` 表示操作无需等待（空传输，或 Ascend 950 上的同步 SDMA 路径），此时 `Wait` 和 `Test` 立即返回 `true`；`valid()` 仅反映 `handle != 0`，不能用于判断成功与否。

`CompletionRecordCount` 在 `handle == 0` 时返回 0；非零 handle 无法解析时返回 `kCompletionRecordCountError`（`UINT32_MAX`），该值须按错误处理。`CompletionRecordAt` 无法解析时返回 `addr == nullptr` 的空记录。

### AsyncCompletionRecord

完成记录描述一次传输的完成判据，供无法执行设备侧轮询原语的消费者使用。消费者按 `kind` 比较 `addr` 处的值与 `expected`，无需了解后端队列布局：

```cpp
enum class CompletionKind : uint32_t {
    SDMA_POST_DONE = 1,
    URMA_CQE_DW0 = 2,
    RDMA_HNS1825_CQE = 3,
};

struct AsyncCompletionRecord {
    __gm__ uint8_t *addr;
    uint64_t expected;
    CompletionKind kind;
};
```

### AsyncSession

用于异步 DMA 操作的引擎无关会话对象，构建一次后传递给所有异步调用：

```cpp
comm::AsyncSession session;
comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, workspace, session);
session.submitMode = comm::AsyncSubmitMode::DEFER;  // 可选，默认 IMMEDIATE 逐条提交
session.batchSize = 8U;                             // 可选，批量提交阈值
```

定义于 `include/pto/comm/async_common/async_types.hpp`。构建参数和批量提交详见 [TPUT_ASYNC](TPUT_ASYNC_zh.md)。

### ParallelGroup

用于多 NPU 集合通信的包装器：

```cpp
template <typename GlobalData>
struct ParallelGroup {
    // 指向 `GlobalData` 对象数组的指针（每个对象封装一个 GM 地址）。
    // 数组本身是本地元数据；封装的地址可以指向本地或远端 GM，
    // 具体取决于集合通信指令的语义。
    GlobalData *tensors;
    int nranks;   // rank 总数
    int rootIdx;  // 根 NPU 的 rank 索引

    // 工厂函数（推荐）：从已有 tensor 数组构建。
    static ParallelGroup Create(GlobalData *tensorArray, int size, int rank_id);
};
```
