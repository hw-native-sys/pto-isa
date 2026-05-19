# Cube 缓冲层级

Cube core（AIC）操作的是一个独立于 Vector 块 UB 的专用缓冲层级。Cube 操作数依次经过 `L1`（cbuf）→ `L0A` / `L0B` → `L0C` → 回写，可选辅助缓冲为 `BT`（bias table）与 `FB`（FIXPIPE buffer）。

## 地址空间

| 空间 | 角色 | 布局 | 典型生产者 | 典型消费者 |
|------|------|------|------------|------------|
| `gm`  | Global Memory（片外 HBM/DDR） | ND 行优先 | host / kernel | DMA 加载器 |
| `l1`  | Cube CBUF，片上约 1 MB | NZ fractal | `pto.mte_gm_l1`、`pto.mte_gm_l1_frac`、`pto.mte_ub_l1` | `pto.mte_l1_l0a`、`pto.mte_l1_l0b`、`pto.mte_l1_ub`、`pto.mte_l1_bt` |
| `l0a` | Cube 左操作数暂存区 | FRACTAL_NZ（A5）/ FRACTAL_ZZ（A3） | `pto.mte_l1_l0a` | `pto.mad*` |
| `l0b` | Cube 右操作数暂存区 | FRACTAL_ZN（K 最内） | `pto.mte_l1_l0b` | `pto.mad*` |
| `l0c` | Cube 累加器 | MMAD 输出的 FRACTAL_NZ | `pto.mad*` | FIXPIPE 回写（`pto.mte_l0c_*`） |
| `bt`  | Bias Table | 与元素类型匹配的向量 | `pto.mte_l1_bt` | `pto.mad_bias`、`pto.mad_mx_bias` |
| `fb`  | FIXPIPE 辅助缓冲 | 实现相关 | `pto.mte_l1_fb` | FIXPIPE 回写指令 |
| `ub`  | Vector Unified Buffer | ND | DMA 加载器 | vector 流水线 |

各缓冲的精确 NZ 索引顺序见 [NZ Fractal 布局](./nz-fractal-layout_zh.md)。

## 数据流契约

```text
                +----------------- AIC 发射队列 -----------------+
                |    MTE2     MTE1    CUBE (MMAD)    FIXP        |
                |     |         |        |             |         |
GM (ND) --- pto.mte_gm_l1 / pto.mte_gm_l1_frac        |          |
              |   |                                              |
              v   v                                              |
              L1 (NZ) <-- pto.mte_ub_l1 --- UB                   |
              |                                                  |
       +------+-----+---------------------+                      |
       |            |                     |                      |
   mte_l1_l0a   mte_l1_l0b           mte_l1_bt / mte_l1_fb        |
       |            |                     |                      |
       v            v                     |                      |
      L0A          L0B                    |                      |
       |            |                     |                      |
       +-----+------+                     |                      |
             |                            |                      |
             |     pto.mad / pto.mad_acc / pto.mad_bias / *_mx*   |
             |     <----------------------+                      |
             v                                                   |
            L0C                                                  |
             |                                                   |
             +-- pto.mte_l0c_l1 / pto.mte_l0c_gm / pto.mte_l0c_ub +
                  （FIXPIPE 回写）
```

## 对齐与尺寸约定

- 所有 cube 缓冲指针（L1 / L0A / L0B / L0C / BT / FB）都要求 32 字节对齐。
- L0A 与 L0B 的 fractal tile 是 512B（一个 32B 宽 × 16 行的 block，按相应的内层朝向）。
- L0C 累加器 tile 使用 `N1 M1 M0 N0` 顺序，方便 FIXPIPE 每次流式输出一行 M 维结果。
- 按元素类型派生的内层宽度（`K0 = N0 = C0 / sizeof(T)`）遵循 [NZ Fractal 布局](./nz-fractal-layout_zh.md)。

## 同步

Cube 程序由 AIC 的 Scalar Unit（SU）发射到 MTE2 / MTE1 / CUBE / FIXP 各自的发射队列。与 Vector 块的同步通过 System Controller（SC）的信号量、以及专用 1:2 fixpipe 广播路径来实现。详见：

- [流水线同步](../scalar/ops/pipeline-sync/)：用于在 AIC 内对 MTE2 → MTE1 → CUBE → FIXP 排序的 `pto.set_flag` / `pto.wait_flag` 原语。
- [Cluster 编程模型](../machine-model/execution-agents_zh.md)：AIC 与 AIV 之间使用的跨块原语（`pto.set_intra_block` / `pto.wait_intra_core`）。

## 相关章节

- [NZ Fractal 布局](./nz-fractal-layout_zh.md)
- [FIXPIPE 模型](./fixpipe-model_zh.md)
- [Cube 数据搬运指令](./README_zh.md#cube-数据搬运指令)
