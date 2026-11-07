# TMATMUL_MX HiF4（HiFloat4 Cube矩阵乘）

## 简介

使用 `hifloat4x2_t` 类型的 `TMATMUL_MX` 执行 Cube 矩阵乘法，A 和 B 操作数均为
HiF4（4位）压缩数据，各自附带三级 HiF4 缩放因子（Ea/Eb/Ec）。结果在 L0C 中以
FP32 累加，通常通过 FIXPIPE 转换为 BF16 后存储。

Ascend 960（dav-920r1）HiF4 矩阵乘流水线：
**TLOAD → TEXTRACT → TMATMUL_MX → TSTORE（FIXPIPE）**。

## 流水线

```
GM (BF16) ──TLOAD──▶ L1 ──TEXTRACT──▶ L0A/L0B + L0AMX/L0BMX ──TMATMUL_MX──▶ L0C (FP32) ──TSTORE──▶ GM (BF16)
```

- **TLOAD**：加载 HiF4 数据（GM→L1）和缩放字节（GM→L1，HIF4_A_ZZ / HIF4_B_NN 布局）。
  数据使用 `copy_gm_to_cbuf_multi_nd2nz`（ND→NZ 分形）。HiF4 缩放字节还支持
  `HIF4_SCALE_ND → HIF4_A_ZZ` 和 `HIF4_SCALE_DN → HIF4_B_NN` 两种搬运模式。
- **TEXTRACT**：通过 `load_c_buf_to_ca_s4` 将数据从 L1 移至 L0A/L0B，通过
  `load_c_buf_to_ca_mx` 将缩放因子移至 L0AMX/L0BMX。
- **TMATMUL_MX**：Cube 单元的 `mad_mx`，使用 `hifloat4x2_t` 类型标记，内部触发
  三级 Ea/Eb/Ec 缩放因子应用。
- **TSTORE**：FIXPIPE 将 FP32 转换为 BF16 并写入 GM。

## 缩放布局（[16,4] = 64B 单元）

每个 HiF4 缩放补丁覆盖一个 M 或 N 分形 × 一个 K 组（64个元素）。补丁大小为 64 字节：

```
字节  0..31：[Ea(g0), Eb(g0), Ea(g1), Eb(g1), ... × 16 组]  (EaEb 半区)
字节 32..63：[Ec_lo(g0), Ec_hi(g0), ... × 16 组]            (Ec 半区)
```

- **Ea**：8位 e6m2 指数（每64元素组）。
- **Eb**：8位打包（每8元素子组一个），保留全部8位。
- **Ec**：16位打包（每4元素子组一个）。

CCE TQuant 通过 `pstu`（谓词→对齐寄存器）+ `vstas`（对齐→UB）存储 Eb，以输出频率
打包谓词，避免了旧的 `DS_B8` 降采样丢失 Eb 位 b4–b7 的问题。

## TLOAD 缩放字节布局

TMATMUL_MX HiF4 中，A/B 缩放字节在 L1 中仍使用 HIF4_A_ZZ / HIF4_B_NN 分形布局；
GM 侧支持两种形态：

- **分形 GM**：GM 本身即为 HIF4_A_ZZ / HIF4_B_NN 布局的 `uint8_t` 数据，
  TLOAD 直接搬运到 L1。
- **2D GM**：GM 为普通行主序二维张量，每个元素是 `hif4_scale`（4 字节，打包
  Ea/Eb/Ec 中的 4 个字节）。A 侧布局为 `HIF4_SCALE_ND`，形状 [M, K/64]；
  B 侧布局为 `HIF4_SCALE_DN`，形状 [K/64, N]。TLOAD 将每个 `hif4_scale`
  展开成 4 个 L1 `uint8_t`，并分别完成 `HIF4_SCALE_ND → HIF4_A_ZZ` 与
  `HIF4_SCALE_DN → HIF4_B_NN` 的分形搬运。

2D GM 模式适用于上游已按逻辑矩阵生成缩放数据、尚未在 GM 中展开为 HIF4 分形
布局的场景。数据搬运后，TEXTRACT 与 TMATMUL_MX 仍读取 L1 中统一的
HIF4_A_ZZ / HIF4_B_NN 缩放布局，因此不会影响后续流水线的接口约定。

约束如下：

- GM 的 `RawDType` 必须为 `hif4_scale`；L1 缩放 tile 的 `DType` 必须为 `uint8_t`。
- `HIF4_SCALE_ND` 的 M 维与 `HIF4_SCALE_DN` 的 N 维必须为 16 对齐。
- A 侧 L1 tile 有效列数等于 GM 列数 × 4；B 侧 L1 tile 有效行数等于 GM 行数 × 4。
- L1 缩放 tile 的 `SFractalSize` 必须为 32，且容量不得超过 512 KB。

## L0C 容量与 N 分块

L0C 为 256 KB，以 FP32（4字节/元素）累加。当 `M × N × 4 > 256 KB` 时，内核按 N 分块：

- `tileN = floor(L0C_SIZE / (M × 4))`，向下取整到 64（TEXTRACT 列对齐）。
- A 侧（跨 N 分块共享）在循环前加载并提取一次。
- 每次迭代提取 `tileN` 宽的 B 列切片，执行 `TMATMUL_MX`，将 `M × tileN` 块以
  偏移 `j × tileN`、步幅 `N` 存入 GM。

## 累加器数据类型

L0C 累加器始终为 `float`（4字节），由 `CheckMadMxValid` 强制：
`static_assert(Rows × Cols × sizeof(float) <= PTO_L0C_SIZE_BYTES)`。

## 测试用例

| 测试用例 | 形状 | 用途 |
|---|---|---|
| `tmatmul_mx_hif4` | 128×128×128, 128×256×128, 256×128×128, 64×64×64, 256×256×256, 128×512×128, 512×128×512, 128×128×256, 256×128×512 | HiF4 Cube 矩阵乘端到端 |
| `tmatmul_mx_hif4_nd2zz_dn2nn` | 128×128×128, 128×256×128, 256×128×128, 128×128×256 | A/B 缩放字节走 2D GM `HIF4_SCALE_ND/DN` 加载 |
| `tmatmul_mx_e1m2` | 128×128×128 | e1m2 MX 参考用例（相同流水线，普通 MX 缩放） |

## 参考

- HiF4 标准：`docs/isa/TQUANT_HIF4.md`，arXiv:2602.11287
- CCE 实现：`include/pto/npu/a6/TQuant.hpp`、`TMatmul.hpp`、`TLoad.hpp`、`TExtract.hpp`
