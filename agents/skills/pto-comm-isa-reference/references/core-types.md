# 核心类型详解

## Signal — 标量信号

用于单标志同步，封装 `int32_t` 类型的 GM 地址：

```cpp
using Signal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

comm::Signal sig(ptr);  // ptr: __gm__ int32_t*
```

## Signal2D — 二维信号矩阵

编译期形状的二维信号网格，支持密集布局和子区域视图：

```cpp
// 密集 4×8 网格（步长自动推导为 8）
comm::Signal2D<4, 8> grid(ptr);

// 从 128 列大网格中的子区域（步长 = 128）
comm::Signal2D<4, 8> sub(ptr + offset, 128);
```

## ParallelGroup — 集合通信分组

轻量级视图，封装多 rank 的 `GlobalTensor` 对象数组：

```cpp
template <typename GlobalData>
struct ParallelGroup {
    GlobalData *tensors;   // 每个 rank 的 GlobalTensor 数组
    int nranks;            // rank 总数
    int rootIdx;           // root NPU 的 rank 索引

    static ParallelGroup Create(GlobalData *tensorArray, int size, int rootIdx);
};
```

**关键约束**：
- `tensors` 指向外部数组（不做动态内存分配）
- `rootIdx` 是 root rank 在组中的索引，所有 rank 必须传入相同的 `rootIdx`
- 通过 `operator[]` 按 team rank 索引访问

## NotifyOp — 通知操作类型

| 值 | 说明 |
|----|------|
| `NotifyOp::AtomicAdd` | 原子加（`signal += value`） |
| `NotifyOp::Set` | 直接赋值（`signal = value`） |

## WaitCmp — 比较运算符

| 值 | 说明 |
|----|------|
| `WaitCmp::EQ` | 等于 (`==`) |
| `WaitCmp::NE` | 不等于 (`!=`) |
| `WaitCmp::GT` | 大于 (`>`) |
| `WaitCmp::GE` | 大于等于 (`>=`) |
| `WaitCmp::LT` | 小于 (`<`) |
| `WaitCmp::LE` | 小于等于 (`<=`) |

## ReduceOp — 归约运算符

| 值 | 说明 |
|----|------|
| `ReduceOp::Sum` | 逐元素求和 |
| `ReduceOp::Max` | 逐元素取最大值 |
| `ReduceOp::Min` | 逐元素取最小值 |

## AtomicType — 原子操作类型

定义于 `include/pto/common/constants.hpp`：

| 值 | 说明 |
|----|------|
| `AtomicType::AtomicNone` | 无原子操作（默认） |
| `AtomicType::AtomicAdd` | 原子加操作 |

## DmaEngine — DMA 引擎选择

| 值 | 说明 |
|----|------|
| `DmaEngine::SDMA` | SDMA 引擎，支持二维传输 |
| `DmaEngine::URMA` | URMA 引擎，支持一维传输（仅 Ascend950 / NPU_ARCH 3510） |

## AsyncEvent — 异步事件句柄

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                           // handle != 0 时返回 true
    bool Wait(const AsyncSession &session) const; // 阻塞直到传输完成
    bool Test(const AsyncSession &session) const; // 非阻塞完成检测
};
```

## AsyncSession — 异步会话

引擎无关的会话对象，通过 `BuildAsyncSession<engine>()` 构建：

```cpp
struct AsyncSession {
    DmaEngine engine;
    sdma::SdmaSession sdmaSession;
    urma::UrmaSession urmaSession;
    bool valid;
};
```
