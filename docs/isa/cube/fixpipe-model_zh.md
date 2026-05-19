# FIXPIPE 回写模型

`FIXPIPE` 是 cube 专用的回写 / 后处理流水线。它把 `L0C` 累加器结果搬到 `L1`、`UB` 或 `GM`，同时把目标所要求的布局变换（NZ → ND）一起做完，并可选地附加反量化 / scale / clip / 激活等后处理。

本页描述以下三条回写指令共享的 FIXPIPE 地址模型：

- [`pto.mte_l0c_l1`](./ops/data-movement/mte-l0c-l1_zh.md)：L0C → L1
- [`pto.mte_l0c_gm`](./ops/data-movement/mte-l0c-gm_zh.md)：L0C → GM
- [`pto.mte_l0c_ub`](./ops/data-movement/mte-l0c-ub_zh.md)：L0C → UB

## 源布局

L0C 源 tile 布局为 `N1 M1 M0 N0`（FRACTAL_NZ：外层列优先、内层行优先）。FIXPIPE 每次寻址一行 M 维结果，并按目标内存的自然顺序输出。

## 回写阶段的 NZ → ND 转换

对 L0C 中的每个 cube 片段，FIXPIPE 应用：

```text
C_nz[n1][m1][m0][n0]  -->  C_nd[m1*M0 + m0][n1*N0 + n0]
```

转换与回写**融合**完成——不需要单独的显式转置。目标 stride 在 FIXPIPE 指令上以 ND 坐标表达。

## 双目标广播（1 → 2 Cube-to-Vector）

当 FIXPIPE 目标是 Vector 块的 UB 时，cube 可以通过专用片上数据通路同时广播到 AIV0 与 AIV1 的 UB 区域，按行轴（`DualModeSplitM`）或列轴（`DualModeSplitN`）切分 tile：

| 切分 | AIV0 接收 | AIV1 接收 |
|------|-----------|-----------|
| Split-M（按行） | ND 中上半 `[M/2, N]` | ND 中下半 `[M/2, N]` |
| Split-N（按列） | ND 中左半 `[M, N/2]` | ND 中右半 `[M, N/2]` |

这种 1→2 在硬件中带 tile 切分的广播，是 1:2 Cube-to-Vector tile 分发的架构基础，在 `pto.mte_l0c_ub` 上通过属性选择。

## Burst / Loop 模型

与 [标量 DMA](../scalar/dma-copy_zh.md) 和 [cube 数据搬运](./README_zh.md#cube-数据搬运指令) 一样，FIXPIPE 回写使用内联 `nburst(...)` / `loop(...)` 子句，无需外部配置寄存器。

## 后处理 hook

FIXPIPE 可以在回写路径上沿途应用以下后处理，通过 clause 或 [`pto.mte_l1_fb`](./ops/data-movement/mte-l1-fb_zh.md) 加载的 `FB` payload 配置：

- 反量化（每通道 scale / zero-point）
- Clip / 饱和到目标元素类型范围
- 激活（ReLU / clipped linear，目标相关）

每条 `pto.mte_l0c_*` 的 per-op 页面会说明它支持哪些后处理 clause。

## FIXPIPE 周围的同步

`FIXP` 是 AIC 侧四条发射队列之一（其余三条是 `MTE2`、`MTE1`、`CUBE`）。标准的生产者 / 消费者链是：

```text
CUBE (pto.mad*)  --set_flag(CUBE -> FIXP)-->  FIXP (pto.mte_l0c_*)  -->  L1 / UB / GM
```

`pto.mad*` 完成一个 L0C tile 后，生产者必须通过其中一个事件 ID 把 `set_flag` 从 `PIPE_CUBE` 发到 `PIPE_FIXP`；FIXPIPE 消费者在对同一个 L0C tile 发出 `pto.mte_l0c_*` 之前必须发出匹配的 `wait_flag`。否则会读到尚未提交的 L0C 状态，verifier 会报错。

## 相关章节

- [NZ Fractal 布局](./nz-fractal-layout_zh.md)
- [缓冲层级](./buffer-hierarchy_zh.md)
- [流水线同步](../scalar/ops/pipeline-sync/)
