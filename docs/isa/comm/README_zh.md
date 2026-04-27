# PTO 通信 ISA 参考手册

本目录包含 PTO 通信 ISA 的逐指令参考文档。

- 权威来源（C++ 内建接口）：`include/pto/comm/pto_comm_inst.hpp`
- 类型定义：`include/pto/comm/comm_types.hpp`

## 点对点通信（同步）
- [**TPUT**](TPUT.md)：远程写（GM → UB → GM）
- [**TGET**](TGET.md)：远程读（GM → UB → GM）

## 点对点通信（异步）
- [**TPUT_ASYNC**](TPUT_ASYNC.md)：异步远程写（GM → DMA 引擎 → GM）
- [**TGET_ASYNC**](TGET_ASYNC.md)：异步远程读（GM → DMA 引擎 → GM）

## 基于信号的同步
- [**TNOTIFY**](TNOTIFY.md)：向远端 NPU 发送通知
- [**TWAIT**](TWAIT.md)：阻塞等待信号条件满足
- [**TTEST**](TTEST.md)：非阻塞检测信号条件

## 集合通信

- [**TGATHER**](TGATHER.md)：从所有 rank 收集数据
- [**TSCATTER**](TSCATTER.md)：向所有 rank 分发数据
- [**TREDUCE**](TREDUCE.md)：从所有 rank 归约数据到本地
- [**TBROADCAST**](TBROADCAST.md)：从当前 NPU 广播数据到所有 rank

## 类型定义

### NotifyOp

`TNOTIFY` 的操作类型：

| | 值 | 说明 |
|-|-------|-------------|
| | `NotifyOp::Set` | 直接赋值（`signal = value`）|
| | `NotifyOp::AtomicAdd` | 原子加（`signal += value`）|

### WaitCmp

`TWAIT` 和 `TTEST` 的比较运算符：

| | 值 | 说明 |
|-|-------|-------------|
| | `WaitCmp::EQ` | 等于（`==`）|
| | `WaitCmp::NE` | 不等于（`!=`）|
| | `WaitCmp::GT` | 大于（`>`）|
| | `WaitCmp::GE` | 大于等于（`>=`）|
| | `WaitCmp::LT` | 小于（`<`）|
| | `WaitCmp::LE` | 小于等于（`<=`）|

```cpp
// 用法示例（统一运行时参数风格）：
comm::NOTIFY(signal, 1, comm::NotifyOp::Set);
comm::WAIT(signal, 1, comm::WaitCmp::EQ);
comm::TEST(signal, 1, comm::WaitCmp::GE);
```

### ReduceOp

`TREDUCE` 的归约运算符：

| | 值 | 说明 |
|-|-------|-------------|
| | `ReduceOp::Sum` | 逐元素求和 |
| | `ReduceOp::Max` | 逐元素取最大值 |
| | `ReduceOp::Min` | 逐元素取最小值 |

### AtomicType

`TPUT` 的原子操作类型（定义于 `include/pto/common/constants.hpp`）：

| | 值 | 说明 |
|-|-------|-------------|
| | `AtomicType::AtomicNone` | 无原子操作（默认）|
| | `AtomicType::AtomicAdd` | 原子加操作 |

### DmaEngine

`TPUT_ASYNC` 和 `TGET_ASYNC` 的 DMA 后端选择：

| | 值 | 说明 |
|-|-------|-------------|
| | `DmaEngine::SDMA` | SDMA 引擎（支持二维传输）|
| | `DmaEngine::URMA` | URMA 引擎（支持一维传输，仅 Ascend950 / NPU_ARCH 3510）|

### AsyncEvent

由 `TPUT_ASYNC` / `TGET_ASYNC` 返回，用于同步传输完成状态：

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                        // handle != 0 时返回 true
    bool Wait(const AsyncSession &session) const; // 阻塞直到传输完成
    bool Test(const AsyncSession &session) const; // 非阻塞完成检测
};
```

### AsyncSession

用于异步 DMA 操作的引擎无关会话对象，构建一次后传递给所有异步调用：

```cpp
comm::AsyncSession session;
comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, workspace, session);
```

定义于 `include/pto/comm/async/async_types.hpp`。构建参数详见 [TPUT_ASYNC](TPUT_ASYNC.md)。

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
