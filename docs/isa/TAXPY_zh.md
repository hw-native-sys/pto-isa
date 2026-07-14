# TAXPY

## 简介

对 Tile 执行原位缩放累加（AXPY，$a \cdot x + y$）：将 `src0` 按标量 `scalar` 缩放后累加到 `dst` 上。

$$ \mathrm{dst}_{i,j} \leftarrow \mathrm{scalar} \cdot \mathrm{src0}_{i,j} + \mathrm{dst}_{i,j} $$

`dst` 既是累加输入（$y$）也是输出，调用前必须已初始化；`src0`（$x$）只读；`scalar`（$a$）为标量。

## 数学语义

对于有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j}^{\text{new}} = \mathrm{scalar} \cdot \mathrm{src0}_{i,j} + \mathrm{dst}_{i,j}^{\text{old}} $$

- `dst`：读-修改-写（RMW）。读入旧值作为累加基 $y$，写回 $\mathrm{scalar} \cdot x + y$。
- `src0`：只读，逐元素参与运算（$x$）。
- `scalar`：标量缩放系数（$a$），类型为 `TileDataSrc::DType`。

> 除非另有说明，语义在有效区域内定义，目标相关行为标记为实现定义。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TAXPY(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &...events);
```

| 参数 | 方向 | 含义 |
|------|------|------|
| `dst` | 输入/输出 | 累加基与结果 Tile（$y$），读-修改-写，`Vec` |
| `src0` | 输入 | 缩放源 Tile（$x$），只读，`Vec`，有效形状与 `dst` 相同 |
| `scalar` | 输入 | 标量缩放系数（$a$），类型为 `TileDataSrc::DType` |
| `events...` | 输入 | 等待事件（`WaitEvents`），指令前隐式 `TSYNC` |

## Tile 尺寸与数据类型

对于有效形状 $M \times N$：

| Tile | dtype | 有效形状 | TileType | 说明 |
|------|-------|---------|----------|------|
| `dst` | `half` 或 `float` | $M \times N$ | `Vec` (UB) | 累加基 + 结果（RMW） |
| `src0` | `half` 或 `float` | $M \times N$ | `Vec` (UB) | 缩放源，逐元素 |

> `dst` 与 `src0` 的有效行数、有效列数必须完全相同。

## 支持的输入 dtype

| `dst` dtype | `src0` dtype | `scalar` dtype | 说明 |
|-------------|--------------|----------------|------|
| `half` | `half` | `half` | 同类型路径，直接 `vaxpy` |
| `float` | `float` | `float` | 同类型路径，直接 `vaxpy` |
| `float` | `half` | `half` | 差异路径：`src0` 拓宽为 FP32 后累加 |

> `dst` 与 `src0` 必须 dtype 一致，或 `dst` 为 `float` 且 `src0` 为 `half`（允许 half→float 的拓宽累加）。`dst` 为 `half` 而 `src0` 为 `float` 的组合非法（实现内 `static_assert` 拦截）。

## 实现说明

TAXPY 在向量流水线（`PIPE_V`）上执行，使用 `vaxpy`（$a \cdot x + y$）向量内建：

1. **同类型（`dst` 与 `src0` 同 dtype）**：逐 repeat 加载 `src0` 与 `dst`，执行 `vaxpy(dst, src0, scalar)` 后写回 `dst`；尾部不足一个 repeat 的列由谓词掩码屏蔽。
2. **差异类型（`dst`=`float`，`src0`=`half`）**：`src0` 的 half 数据拓宽为 FP32 后参与累加（A5 上经 `UNPK_B16` 解包并 `vcvt` 转换；A2/A3 由 `vaxpy` 原生按 4-block src / 8-block dst 处理）。
3. A2/A3 上按 repeat-stride 是否溢出、以及列数与行数的关系，在 count 模式与 norm 模式间选择，以覆盖任意有效形状。

## 约束

| 约束 | 适用范围 | 原因 |
|------|---------|------|
| `dst`、`src0` 必须为 `TileType::Vec` | 所有目标 | 在 UB（向量流水线）上执行 |
| `dst` 与 `src0` 有效形状相同（$M \times N$） | 所有目标 | 逐元素一一对应 |
| `dst` dtype ∈ {`half`, `float`} | 所有目标 | `vaxpy` 支持的浮点字宽 |
| `dst`/`src0` dtype 一致，或 (`float`,`half`) | 所有目标 | 仅允许 half→float 拓宽累加 |
| `dst` 调用前必须已初始化 | 所有目标 | `dst` 作为累加基 $y$ 被读入 |

## 示例

```cpp
// dst 必须先初始化（作为累加基 y）；结果：dst = scalar * src0 + dst
TAXPY(dstTile, srcTile, scalar);
```

完整 ST 示例见 `tests/npu/a5/src/st/testcase/taxpy/`（A5）、`tests/npu/a2a3/src/st/testcase/taxpy/`（A2/A3）、`tests/npu/kirin9030/src/st/testcase/taxpy/`（Kirin9030）及 `tests/cpu/st/testcase/taxpy/`（CPU 参考实现）。
