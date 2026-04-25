# 内存优化

本文介绍 PTO 内核的内存优化技巧。片上内存抽象模型参见 [`docs/machine/abstract-machine.md`](../machine/abstract-machine.md)；示例驱动的调优参见 [`opt.md`](opt.md)。

---

## 1. 片上存储模型

PTO 通过 Tile 类型暴露片上存储，每种 `TileType` 对应一类存储区域：

| Tile 类型 | 存储区 | 典型用途 |
|-----------|--------|----------|
| `TileType::Vec` | 向量缓冲区（UB） | 逐元素操作、规约 |
| `TileType::Mat` | 矩阵暂存区（类 L1） | TLOAD 目标、GEMM 暂存 |
| `TileType::Left/Right` | 矩阵操作数寄存器（L0A/L0B） | TMATMUL 输入 |
| `TileType::Acc` | 累加器寄存器（L0C） | TMATMUL 输出 |

数据流向固定：

```
GM --TLOAD--> Mat --TMOV/TEXTRACT--> Left/Right --TMATMUL--> Acc --TSTORE--> GM
```

向量操作（`TADD`、`TEXP`、`TROWSUM` 等）直接作用于 `Vec` Tile。
`Mat` 与 `Vec` 之间的转换使用 `TMOV` 或 `TEXTRACT`。

---

## 2. Tile 尺寸与容量

### 声明 Tile 前先估算占用

```cpp
constexpr int TM = 128, TK = 64, TN = 256;
// 暂存区（双缓冲）
constexpr size_t staging = 2*(TM*TK + TK*TN)*sizeof(half);  // 2*(16+32) KB = 96 KB
// 累加器
constexpr size_t accum   = TM * TN * sizeof(float);          // 128 KB
// 合计：224 KB——须在目标平台片上容量限制内
```

如果超出限制：缩小 Tile 尺寸、改为单缓冲，或拆分累加器范围。

### 对齐规则（由 `static_assert` 编译期检查）

- 行主序 Tile：`Cols * sizeof(Element)` 必须是 32 字节的整数倍。
- 列主序 Tile：`Rows * sizeof(Element)` 必须是 32 字节的整数倍。
- Boxed Tile（`TileLeft`、`TileRight`、`TileAcc`）：形状须匹配分形基础 Tile 尺寸（`fractalABSize=512`，`fractalCSize=1024`）。

`fractalABSize=512` 下常见基础形状：

| 元素类型 | 基础 rows×cols |
|----------|---------------|
| `fp32`   | 16×8 |
| `fp16`   | 16×16 |
| `int8`   | 16×32 |

---

## 3. 数据复用

### K 维度分块（GEMM）

每个 A/B 面板只加载一次，在完整 K 循环内累加：

```cpp
TileAcc<float, TM, TN> acc;
TFILL(acc, 0.0f);

for (int k = 0; k < K; k += TK) {
    Tile<TileType::Mat, half, TM, TK> a_mat;
    Tile<TileType::Mat, half, TK, TN> b_mat;
    TLOAD(a_mat, gA_view);    // 每个 TM×TK 块只访问一次 GM
    TLOAD(b_mat, gB_view);

    TileLeft<half, TM, TK>   a_left;
    TileRight<half, TK, TN>  b_right;
    TMOV(a_left, a_mat);
    TMOV(b_right, b_mat);

    TMATMUL_ACC(acc, a_left, b_right);  // acc 全程驻留片上
}
TSTORE(gC_view, acc);   // 循环结束后一次性写回
```

### 缓存行统计量（Softmax）

```cpp
Tile<TileType::Vec, float, R, C> input, shifted, exp_v, output;
Tile<TileType::Vec, float, R, 1> row_max, row_sum;

TLOAD(input, gInput);
TROWMAX(row_max, input);             // 计算一次，驻留片上
TROWEXPANDSUB(shifted, input, row_max);
TEXP(exp_v, shifted);
TROWSUM(row_sum, exp_v);             // 计算一次，驻留片上
TROWEXPANDDIV(output, exp_v, row_sum);
TSTORE(gOutput, output);
```

若将 `row_max`/`row_sum` 写回再重新加载，GM 流量约增加三倍。

---

## 4. 布局与对齐

**让布局与消费指令匹配**，避免隐式转换。

- GM 数据为行主序时，以 `BLayout::RowMajor` 加载 `Mat` Tile。
- 在支持的目标上，对 `GlobalTensor` 使用 `Layout::NZ` 可消除 GEMM 输入的 `TMOV` 阶段。
- 只有源布局与目标布局确实不同时才使用 `TTRANS`。

`TileLeft` / `TileRight` / `TileAcc` 别名会自动选定正确的 `SLayout` 和 `SFractalSize`：

```cpp
TileLeft<half, 128, 64>   a_left;   // 外层列主序 + 内层行主序
TileRight<half, 64, 256>  b_right;  // 外层行主序 + 内层列主序
TileAcc<float, 128, 256>  acc;
```

没有明确理由，不要手动覆盖 `SLayout` 或 `SFractalSize`。

---

## 5. 减少全局内存流量

### 算子融合

将连续操作融合到同一内核，消除中间结果的 GM 存储和重新加载：

```cpp
// 未融合：4 次 GM 往返
TLOAD(a, gInput); TADD(b, a, s); TSTORE(gTmp, b);
TLOAD(c, gTmp);   TMUL(d, c, s2); TSTORE(gOut, d);

// 融合后：2 次 GM 往返
TLOAD(a, gInput);
TADD(b, a, s);    // b 驻留片上
TMUL(d, b, s2);
TSTORE(gOut, d);
```

### 连续访问模式

优先使用行主序遍历；列主序跨步访问会降低突发传输效率。
若必须按列访问，可加载行主序块后在片上 `TTRANS`。

### TPREFETCH

发出非阻塞提示，将数据移动与计算重叠：

```cpp
if (k + TK < K) { TPREFETCH(gA_next); TPREFETCH(gB_next); }
TMATMUL_ACC(acc, a_left, b_right);  // 与预取并行执行
```

---

## 6. 双缓冲

在两组暂存 Tile 之间交替，使内存流水线与计算流水线同时保持繁忙：

```cpp
Tile<TileType::Mat, half, TM, TK> a_mat[2];
Tile<TileType::Mat, half, TK, TN> b_mat[2];
TileLeft<half, TM, TK>            a_left[2];
TileRight<half, TK, TN>           b_right[2];

Event<Op::TLOAD, Op::TMOV>    ev_load[2];
Event<Op::TMOV,  Op::TMATMUL> ev_mov[2];

// 预热
ev_load[0] = TLOAD(a_mat[0], gA_view_0);
ev_load[0] = TLOAD(b_mat[0], gB_view_0);

for (int k = 0, ping = 0; k < K; k += TK, ping ^= 1) {
    int pong = ping ^ 1;
    // 加载下一个（pong）的同时计算当前（ping）
    if (k + TK < K) {
        ev_load[pong] = TLOAD(a_mat[pong], gA_view_next);
        ev_load[pong] = TLOAD(b_mat[pong], gB_view_next);
    }
    ev_mov[ping]  = TMOV(a_left[ping],  a_mat[ping],  ev_load[ping]);
    ev_mov[ping]  = TMOV(b_right[ping], b_mat[ping],  ev_load[ping]);
    TMATMUL_ACC(acc, a_left[ping], b_right[ping], ev_mov[ping]);
}
TSTORE(gC_view, acc);
```

稳定运行时，实际吞吐量趋近于 `max(T_load, T_compute)`，而非 `T_load + T_compute`。

事件模型详见 [`Event_zh.md`](Event_zh.md)。

---

## 7. 有效区域与 Padding

当维度不是 Tile 尺寸整数倍时，使用有效区域而非申请多种 Tile 尺寸：

```cpp
// 静态有效区域
using TileT = pto::Tile<pto::TileType::Vec, float, 128, 256,
                        pto::BLayout::RowMajor, 100, 200>;

// 动态有效区域
using TileD = pto::Tile<pto::TileType::Vec, float, 128, 256,
                        pto::BLayout::RowMajor,
                        pto::DYNAMIC, pto::DYNAMIC>;
TileD t(actual_rows, actual_cols);

// 容量 Padding 到对齐边界，有效区域表达实际范围
constexpr int PADDED = ((127*4+31)/32)*32/4;  // 127 列 -> 128
using TileP = pto::Tile<pto::TileType::Vec, float, 16, PADDED,
                        pto::BLayout::RowMajor, 16, 127>;
```

---

## 8. 实践检查清单

**片上容量**
- [ ] 所有 Tile 的片上占用（暂存 + 操作数 + 累加器 × 缓冲区数量）在目标限制内。
- [ ] 双缓冲时，暂存 Tile 数量乘以 2 后再检查。

**对齐**
- [ ] 让 `pto::Tile` 中的 `static_assert` 在编译时捕获违规。
- [ ] 使用 `TileLeft` / `TileRight` / `TileAcc` 别名以保证正确的分形布局。

**数据复用**
- [ ] `TileAcc` 在完整 K 循环期间驻留片上，循环结束后一次性写回。
- [ ] 行/列统计量（`TROWMAX`、`TROWSUM`）缓存在 `Vec` Tile 中，不写回 GM。
- [ ] 内核内无多余的 TSTORE/TLOAD 对。

**GM 流量**
- [ ] 连续逐元素操作融合到同一内核。
- [ ] 行主序 GM 访问；必须列访问时，加载块后片上 `TTRANS`。
- [ ] 使用 `TPREFETCH` 将数据移动与计算重叠。

**同步**
- [ ] 使用 `Event<SrcOp, DstOp>` 进行细粒度排序；稳定循环中不使用全局 `TSYNC`。
- [ ] 每个消费者只等待其对应的生产者事件。

---

## 参考资料

- [抽象机器模型](../machine/abstract-machine.md)
- [Tile 编程模型](Tile_zh.md)
- [GlobalTensor 编程模型](GlobalTensor_zh.md)
- [事件与同步](Event_zh.md)
- [PTO 优化指南](opt_zh.md)
- [流水线与并行执行](pipeline-parallel_zh.md)
- [GEMM 教程](tutorials/gemm_zh.md)
- [GEMM 性能内核](../../kernels/manual/a2a3/gemm_performance/README.md)
- [Flash Attention 内核](../../kernels/manual/common/flash_atten/README.md)
