# 什么是 PTO 虚拟 ISA

PTO ISA（Parallel Tile Operation Instruction Set Architecture）为华为 Ascend NPU 软件定义了一套与具体机器代际无关的虚拟 ISA。它不是任何单一 Ascend 实现的原生二进制指令集，而是位于前端、代码生成器、验证器、模拟器和目标 backend 之间的公共低层契约。

## 为什么是 Tile-First

大多数 Ascend kernel 的实际编写单位是 **tile**，而不是匿名 lane 或不透明 buffer。只用 generic SIMD / SIMT 抽象最终也能描述硬件，但会把真正关键的问题下沉到 backend 私有传说里：

- shape 和 layout 是否合法
- 哪些元素真正有意义（valid region）
- 两个 tile 在什么条件下允许别名
- 同步必须出现在哪一层

PTO 把这些问题直接提升进 ISA，让程序、verifier 和 backend 共享同一套可测试、可移植的契约。

## 两条编译路径

PTO 程序可以通过两条受支持路径进入硬件，两条路径共享同一套 PTO 指令语义。

### Flow A: `ptoas -> C++ -> bisheng -> binary`

高层前端生成 `.pto` 文本，`ptoas` 解析、验证并把它 lowering 成调用 `pto-isa` C++ intrinsic 的代码，再由 `bisheng` 或其他 C++ backend 编译成目标二进制。

```text
High-level Frontend
    -> .pto
    -> ptoas
    -> C++ kernel code
    -> bisheng / backend compiler
    -> binary
```

### Flow B: `ptoas -> binary`

`.pto` 文本也可以直接经由 `ptoas --target=...` 组装为目标二进制，绕过 C++ 中间层。

```text
High-level Frontend
    -> .pto
    -> ptoas --target=a3|a5|cpu
    -> binary
```

## 一个最小例子

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vec_add(Tile<float, 16, 16>& c,
             const GlobalTensor<float>& ga,
             const GlobalTensor<float>& gb) {
    Tile<float, 16, 16> a, b;
    TLOAD(a, ga);
    TLOAD(b, gb);
    TADD(c, a, b);
    TSTORE(gc, c);
}
```

即便这个例子很小，它也已经依赖了 valid region、dtype、layout、显式数据移动和同步等 PTO 的核心概念。

## 关键术语

| 术语 | 含义 |
| --- | --- |
| **PTO** | 围绕 tile、显式数据移动、显式同步和机器可见执行结构建立的编程/指令模型 |
| **PTO ISA** | 本手册定义的虚拟指令集架构 |
| **PTO-AS** | PTO ISA 的文本汇编语法 |
| **ptoas** | 解析 `.pto` 文本并 lowering 到 C++ 或直接生成二进制的工具 |
| **PTOBC** | PTO 程序的字节码分发形式 |
| **Tile** | 带 shape、layout 和 valid-region 元数据的多维片段 |
| **Valid Region** | tile 中真正有架构意义的子区域 |
| **Global Memory (GM)** | 设备侧片外全局内存 |
| **向量 tile buffer** | `TileType::Vec` 使用的本地 tile buffer；当前硬件实现对应 Unified Buffer（UB），但在手册中把它视为同一个 tile-buffer 概念 |
| **Location Intent** | tile 的角色意图，如 `Vec`、`Left`（L0A）、`Right`（L0B）、`Acc`、`ScaleLeft`、`ScaleRight` |
| **Target Profile** | 对 PTO ISA 的具体目标缩窄，例如 CPU、A2A3（Ascend 910B / 910C）、A5（Ascend 950 PR / DT） |

## 软件栈位置

```text
Source Languages / DSLs
        |
        v
   PTO program (.pto)
        |
        +--> ptoas --> C++ --> bisheng --> binary
        |
        +--> ptoas ---------------------> binary
```

这个结构让软件栈在硬件代际变化时仍然共享同一套版本化指令语言。

## 分层指令集结构

PTO ISA 按五类指令集组织：

- `pto.t*`
  Tile 指令集
- `pto.v*`
  向量微指令集
- `pto.*`
  标量与控制指令集
- communication ops
  通信指令
- system scheduling ops
  TPipe/TMPipe 调度与资源生命周期指令

Tile 指令集是主要的编程层；向量指令集提供更细粒度的向量流水控制；标量与控制指令集负责执行外壳；通信指令覆盖跨 rank 行为；系统调度指令覆盖 TPipe/TMPipe 生产者-消费者协议和资源生命周期。

## 相关页面

- [PTO 的设计目标](./goals-of-pto_zh.md)
- [范围与边界](./design-goals-and-boundaries_zh.md)
- [Tile 与有效区域](../programming-model/tiles-and-valid-regions_zh.md)
