# TQUANT HiFloat4（BF16 → HiF4）— 算法与 CCE 映射

## 1. HiFloat4 标准（arXiv:2602.11287）

HiFloat4（HiF4）将 64 个 4 位浮点值与 **32 位**三级共享缩放元数据一起编码。
信息密度为 4.5 位/值，足以直接从压缩 FP4 码驱动 Cube 矩阵乘。

### 1.1 三级层次结构

对于 64 个连续元素的一组，HiF4 计算：

| 层级 | 组大小 | 元数据 | 每元素位数 |
|------|--------|--------|-----------|
| Ea   | 64     | 8位指数（e6m2）| 8/64 = 0.125 |
| Eb   | 8      | 1位指数        | 1/8  = 0.125 |
| Ec   | 4      | 1位指数        | 1/4  = 0.25  |

总元数据：8 + 8×1 + 16×1 = **每64元素32位** = 4.5位/值。

### 1.2 数据元素：FP4 e1m2

每个 4 位数据元素使用 **e1m2** 格式（1符号位，1指数位，2尾数位）：

| 编码 | 符号 | 指数 | 尾数 | 值 |
|------|------|------|------|-----|
| 0000 | 0    | 0    | 00   | +0.0   |
| 0001 | 0    | 0    | 01   | +0.25  |
| 0010 | 0    | 0    | 10   | +0.5   |
| 0011 | 0    | 0    | 11   | +0.75  |
| 0100 | 0    | 1    | 00   | +1.0   |
| 0101 | 0    | 1    | 01   | +1.25  |
| 0110 | 0    | 1    | 10   | +1.5   |
| 0111 | 0    | 1    | 11   | +1.75  |
| 1000 | 1    | 0    | 00   | −0.0   |
| ...  | ...  | ...  | ...  | （负值镜像）|

偏置 = 0，exp=0 时隐式前导 0（非规格化风格），exp=1 时隐式前导 1。可表示范围为 [0, ±1.75]。

### 1.3 量化算法（论文 Algorithm 1）

```
Ma = max(|x|)  per 64-element group
Mb = max(|x|)  per 8-element subgroup
Mc = max(|x|)  per 4-element subgroup

Ea = vcvt_bf162e6m2(Ma, ROUND_R)          // 8位 e6m2 指数
Ea_rec = 1 / Ea                             // 作为 BF16

Eb = (Mb * Ea_rec >= 4) ? 1 : 0            // 每组8元素，1位
Ec = (Mc * Ea_rec * 2^(-Eb) >= 2) ? 1 : 0  // 每组4元素，1位

scale = Ea_rec * 2^(-Eb) * 2^(-Ec)          // 每4元素缩放因子
q = vcvt_bf16_to_e1m2(x * scale)            // 4位量化码
```

## 2. CCE 实现映射（`a6/TQuant.hpp`）

### 2.1 Stage 1: AbsReduceMax（gp4 / gp8 / gp64）

三级树形归约，对 64 个 BF16 输入计算每 4/8/64 元素的绝对值最大值。

### 2.2 Stage 2: CalcExpScale_Cont

读取 Ma/Mb/Mc，产生 Ea/Eb/Ec 指数 + 每 4 元素缩放因子。

```
每循环（128元素 = 8个16元素块）：
  Ma: 8个值（每64元素块1个）   — 加载 E2B_B16（广播到16通道）
  Mb: 64个值（每8元素组1个）   — 加载 US_B16（上采样，2×重复）
  Mc: 128个值（每4元素组1个）  — 加载 NORM

  // Ea: BF16 → e6m2 转换（四舍五入），然后倒数回 BF16
  Ea_e6m2 = vcvt_bf162e6m2(Ma, ROUND_R, PART_EVEN)
  Ea_rec  = vcvt_rcpe6m22bf16(Ea_e6m2, PART_EVEN)    // ≈ 2^(-Ea) 作为 BF16

  // Eb: 阈值检查 — 每8元素最大值是否超过共享指数的4倍？
  Eb_tmp  = Mb * Ea_rec
  Eb_bit  = (Eb_tmp >= 4) ? 1 : 0
  // 通过 pstu（谓词→对齐寄存器）+ vstas（对齐→UB）存储 Eb。
  // 以输出频率打包谓词（每64组1字节，保留全部8个Eb位），
  // 避免了旧的 DS_B8 降采样丢失 b4–b7 位的问题。
  pstu(ureg_Eb, p_Eb, ebPtr); vstas(ureg_Eb, ebPtr, 0, POST_UPDATE)

  // Ec: 阈值检查 — 结合 Eb 校正进一步细化
  Eb_rec  = Eb_bit ? 0.5 : 1.0                        // 2^(-Eb)
  Ec_tmp  = Mc * Ea_rec * Eb_rec
  Ec_bit  = (Ec_tmp >= 2) ? 1 : 0

  // 最终每4元素缩放因子（完整共享指数的倒数）
  Ec_rec  = Ec_bit ? 0.5 : 1.0                        // 2^(-Ec)
  e_scale = Ea_rec * Eb_rec * Ec_rec                  // ≈ 2^(-(Ea+Eb+Ec))
```

**阈值语义：**
- `Mb * Ea_rec >= 4` 表示 `Mb / 2^Ea >= 4`，即 8 元素子块最大值需要多 2 个指数位
  → `Eb = 1` 贡献 `2^1 = 2` 的额外范围。
- `Mc * Ea_rec * Eb_rec >= 2` 表示 `Mc / 2^(Ea+Eb) >= 2`，即 4 元素子块最大值需要
  多 1 个指数位 → `Ec = 1`。

### 2.3 Stage 2b: ExpLayoutForCube（仅 CCE，golden 不包含）

将 Ea/Eb/Ec 从各自的层级扁平布局重排为 Cube 矩阵乘单元消费的交错压缩布局。
Ea 以 `DS_B8` 加载（存储时零扩展）；Eb 以 `NORM` 加载（Stage 2 中的 `pstu`+`vstas`
存储已将其置于输出频率，无需降采样）。
这是硬件特定的数据搬运步骤 — **Python golden 不复制此步骤。**

### 2.4 Stage 3: CalcFp4Values_Cont

```
每循环（256元素）：
  input    = vlds(srcPtr, 128*loop, NORM)             // 128个 BF16 元素
  e_scale  = vlds(scalingPtr, 64*loop, US_B16)        // 每4元素缩放，2×重复
  scaled   = input * e_scale
  fp4_code = vcvt(scaled, ROUND_A, PART_P0)           // BF16 → f4e1m2x2
  vsts(fp4_code, dstPtr, 64*loop, PK4_B32)            // 每字节打包4个半字节
```

### 2.5 e6m2 指数格式

Ea 存储为 8 位 **e6m2** 值（6位指数，2位尾数）。这是一个表示每 64 元素最大值 log2 的
迷你浮点数。`vcvt_bf162e6m2` 四舍五入到最近的 e6m2 可表示值，
`vcvt_rcpe6m22bf16` 计算其作为 BF16 的倒数。

用论文的术语：`Ea = round(log2(Ma))` 量化到 e6m2 精度。

## 3. Python Golden 生成

Python golden 使用 NumPy 复制 Stage 1–3（仅连续情况）：

1. 通过 reshape 计算每 4/8/64 元素绝对值最大值。
2. 通过阈值检查推导 Ea（log2(Ma) 的 e6m2 量化）、Eb、Ec。
3. 计算每 4 元素缩放因子。
4. 将每个 BF16 元素量化为其 e1m2 4 位码。

**不产生 Cube 交错指数布局** — 那是 CCE 特定的数据搬运。Golden 输出：
- `golden_fp4.bin` — 压缩 FP4 e1m2 码（每字节2个）
- `golden_ea.bin` — Ea 指数（每64元素1字节）
- `golden_eb.bin` — Eb 位（每8元素1位，压缩）
- `golden_ec.bin` — Ec 位（每4元素1位，压缩）
- `golden_scale.bin` — 每4元素缩放因子（BF16）

### 3.1 重建误差

侧函数从 FP4 码 + 指数重建原始 BF16 值，报告最大/平均相对误差。
