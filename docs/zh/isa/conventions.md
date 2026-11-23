# PTO ISA 通用约定

本页定义 `docs/zh/isa/` 指令参考文档中使用的通用术语与写法，并与 `include/pto/common/pto_instr.hpp` 中的 C++ 内建接口保持一致。

## 记号

- **Tile**：片上二维操作数对象（例如 `pto::Tile<...>`）。大量指令以 Tile 作为输入/输出，并通过 `GetValidRow()` / `GetValidCol()` 使用 Tile 的有效区域（valid region）。
- **GM（全局内存）**：通过 `pto::GlobalTensor<...>` 访问的片外内存视图。
- **标量 / 立即数**：主机侧标量值，或在 `*S` / `*C` 等变体中编码的立即数参数。

关于这些对象的 C++ 编程模型（类型、布局、枚举、约束等），可参考：

- Tile：`docs/zh/coding/Tile.md`
- GlobalTensor：`docs/zh/coding/GlobalTensor.md`
- 标量与枚举：`docs/zh/coding/Scalar.md`

## 形状与布局

- **行主序 / 列主序**：除非指令页明确声明支持多种布局，否则示例与参考实现默认假设为行主序 Tile。支持多布局的指令会在约束小节中列出具体要求。
- **有效区域（valid region）**：Tile 运行时计算域，通常写作 `(valid_row, valid_col)`，并通过 `GetValidRow()` / `GetValidCol()` 查询。

### 物理形状与有效形状

`Rows` / `Cols` 描述 Tile 的物理存储形状；`GetValidRow()` / `GetValidCol()` 描述参与计算的有效区域。修改有效形状不会改变物理步长。对于非分形 Tile，元素 `(i, j)` 的偏移（单位为元素）为 `i * RowStride + j * ColStride`：RowMajor 的步长是 `(Cols, 1)`，ColMajor 的步长是 `(1, Rows)`。

例如，A5 上 `int64_t` / `uint64_t` 的 RowMajor 输出物理形状为 `[64,4]`、有效形状为 `[64,1]` 时，物理行步长仍为 4 个元素（32 字节）。行归约结果位于偏移 `0, 4, 8, ...`；将有效列数 1 当成物理行步长会导致行间结果错位。紧凑的 ColMajor `[64,1]` 则使用连续的元素偏移。

64 位 Vec Tile 的 32 字节对齐要求作用于物理形状：RowMajor 要求 `Cols % 4 == 0`，ColMajor 要求 `Rows % 4 == 0`。有效行列数可以更小，具体仍须满足指令约束。Tile 分配应覆盖完整物理形状；有效区域之外的填充不等于相邻的独立分配。各指令的填充写入语义见对应文档，尤其是 [TSCATTER](TSCATTER.md)。

### 有效区域语义

在指令页中，当我们写“对有效区域内的每个元素 `(i, j)`”，含义为：

- 除非指令显式定义不同的迭代域，否则默认使用 `valid_row = dst.GetValidRow()`、`valid_col = dst.GetValidCol()`。
- 数学语义仅对 `0 <= i < valid_row` 且 `0 <= j < valid_col` 的 `dst[i, j]` 做出定义。
- 有效区域之外元素的值为**未指定**，除非指令页明确说明（不要假设一定清零或保持不变）。

对多输入指令（例如 `src0`、`src1`），除非约束小节有更严格的要求，文档默认输入 Tile 与迭代域在形状/有效区域上是兼容的。

## 数据类型

每条指令页会列出支持的数据类型（例如 `fp16`、`fp32`、`int8`、`int16`、`int32`、`uint8`、`uint16`、`uint32` 等）。
不同后端/目标对数据类型与布局支持可能不同，具体以对应实现与编译期检查为准。

### 公开枚举

`include/pto/common/type.hpp` 中的下列公开枚举用于参数化指令行为，并出现在内建函数签名中。它们属于公开 API，此处集中说明，以便指令页按名称引用而无需重复罗列取值集合。

- **`VecStoreMode`**（`type.hpp:222`）——选择向量存储的地址布局。`ND`（0）= n 维（连续行主序），`NZ`（1）= 分形 NZ 布局。用于 `TSTORE`/`TSTORE_FP` 中走向量路径存储的重载。
- **`TCopyMode`**（`type.hpp:265`）——选择 Tile 拷贝语义。`SHALLOW_COPY`（0）= 共享底层缓冲区（仅元数据），`DEEP_COPY`（1）= 分配并拷贝缓冲区。
- **`TileLayoutCustom`**（`type.hpp:370`）——自定义布局 Tile 的显式布局标签。取值：`ND`、`DN`、`NZ`、`ZN`、`ZZ`、`NONE`。`NONE` 表示布局由 Tile 构造函数推断；其余取值显式固定布局并绕过推断。
- **`SqrtAlgorithm` / `RsqrtAlgorithm` / `FmodAlgorithm`**（`type.hpp:409` / `:411` / `:419`）——对应数学内建函数的算法选择器。`DEFAULT` = 后端默认精度路径，`HIGH_PRECISION` = 更高精度路径（可能以吞吐换精度）。相同的 `DEFAULT` / `HIGH_PRECISION` 模式由同族数学算法枚举 `PowAlgorithm`、`DivAlgorithm`、`RecipAlgorithm`、`ExpAlgorithm`、`LogAlgorithm`、`FmodSAlgorithm`、`RemAlgorithm`、`RemSAlgorithm`（`type.hpp:405`–`425`）共享；各指令接受哪个选择器参见内建函数签名。

## 事件与同步

- 某些指令序列需要建立内存与向量流水线之间的顺序关系。示例中出现的事件（例如 `set_flag(...)` / `wait_flag(...)`）用于表达后端需要满足的顺序约束。
- 在需要显式同步的场景，使用 event synchronization 建立阶段间的顺序关系。

事件模型可参考：`docs/zh/coding/Event.md`。

## 架构约定索引

部分约定篇幅过长，不便在此内联，集中放在独立页面中。下表每条同时给出约定文档与验证该约定的测试用例，便于开发者从规则一步跳到可运行代码。

| 约定 | 文档 | 验证测试用例 / 来源 |
|-----------|----------|--------------------------|
| A5 mix 跨核 flag 配对（Cube↔Vec、GM/UB 路径、shadow event id） | [a5-cross-core-sync-conventions.md](../../en/isa/a5-cross-core-sync-conventions.md) | `tests/npu/a5/src/st/testcase/tload_mix`、`tests/npu/a5/src/st/testcase/tmov_acc2vec`、`tests/npu/a5/src/st/testcase/tinsert_acc2vec`；实现 `include/pto/npu/a5/custom/TSync_Custom.hpp` |
| A5 片上缓冲区复用屏障（L0A/L0B/L0C/UB 流水线归属、循环复用屏障位置） | [a5-on-chip-storage-reuse.md](../../en/isa/a5-on-chip-storage-reuse.md) | `tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp:286-287`（M→MTE2 屏障）、`:195-196`（M→FIX 屏障） |
| A5 UB 多 Tile 间距与默认基地址 | [a5-on-chip-storage-reuse.md](../../en/isa/a5-on-chip-storage-reuse.md#ub-multi-tile-layout--offset-spacing-and-default-base) | `tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp:234-242`（TASSIGN 块，`0x0` / `0x10000` / `0x20000` 间距）；检查集与默认基地址见 [TASSIGN.md](TASSIGN.md) |
| 错误信息约定（`Fix:` 前缀） | 下文「错误信息」小节 | `include/pto/npu/kirin9030/TGather.hpp:32`（参考示例） |
| 公开 API 命名约定（`T` 前缀） | 下文「公开 API 命名」小节 | `include/pto/common/pto_instr.hpp` |

## 错误信息

因误用 PTO 内建函数而触发的 `static_assert` / `PTO_ASSERT` 消息，必须让开发者**无需再查文档**即可修复调用点。仓库约定是每条此类消息都以 `Fix: ` 前缀开头，并说明四要素：内建函数名、违反的约束、允许的取值集合、以及修复方法。

### 格式

```
Fix: <指令> <哪里错了>；<允许取值 / 修复方法>。
```

- **`Fix: ` 前缀**——消息可执行；开发者在构建日志中搜索 `Fix:` 即可定位需要修改的内容。
- **`<指令>`**——大写的内建函数名（例如 `TGATHER`、`TMATMUL`），便于 `grep` 检索。
- **`<哪里错了>`**——用一句话描述违反的约束（例如 "Dst data type must be ..."）。
- **`<允许取值 / 修复方法>`**——可接受的值或修复动作。

### 参考示例

`include/pto/npu/kirin9030/TGather.hpp:32`：

```cpp
static_assert(..., "Fix: TGATHER Dst data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float.");
```

### 当前状态

`Fix:` 前缀已应用于 `include/pto/npu/{a5,a2a3,kirin9030,kirin9050,kirinX90}`，但 `common` 和 `cpu` 尚未应用。新增针对 PTO 内建函数的 `static_assert` 时，请使用 `Fix:` 前缀和上述四要素形式。将内建函数移植到新后端时，应从参考后端沿用 `Fix:` 消息，而不是重新改写措辞。

## 公开 API 命名

公开 PTO 内建函数遵循 `T` + 操作名的命名约定。`T` 前缀表示 Tile 级原语（区别于运行时/工具符号）。操作名为大写，并与 ISA 页名称一致。

### 约定

- 默认：`T` + 大写操作名——例如 `TADD`、`TMATMUL`、`TSTORE`、`TLOAD`、`TMOV`、`TCOLEXPANDADD`。
- 后缀变体追加简短限定词：`TMOV_FP`（定点搬出）、`TMATMUL_MX`（MX 格式矩阵乘）、`TADDS`（饱和加法）、`TSTORE_FP`（fixpipe 存储）。

### 文档化例外（无 `T` 前缀）

下列公开内建函数**有意**不带 `T` 前缀，因为它们属于控制面/配置原语，而非 Tile 数据变换。此处集中列出，以表明缺失 `T` 前缀是刻意为之而非命名漂移：

| 名称 | 为何无 `T` 前缀 |
|------|-------------------|
| `SYNCALL` | 跨核控制面屏障，不是逐 Tile 操作。 |
| `SETFMATRIX` | Cube 配置（分形矩阵设置），不是 Tile 数据变换。 |
| `SET_QUANT_SCALAR` / `SET_QUANT_VECTOR` | MX 路径的量化 scale 绑定（标量/向量形式），配置原语。 |
| `SET_IMG2COL_*` | Img2col 配置，不是 Tile 数据变换。 |
| `MGATHER` | GM→L1/UB 收集；历史命名无 `T`（UB→UB 形式对应存在 `TGATHER` 数据搬运兄弟指令）。 |
| `MSCATTER` | `MGATHER` 的 GM 分散对应指令；同样为历史命名。 |

新增公开内建函数时，默认使用 `T` 前缀；仅当其为与上表同类别的控制面/配置原语时方可省略，并需在本表添加一条记录。
