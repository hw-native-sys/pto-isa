# THISTOGRAM

> **实现状态**：THISTOGRAM已在Ascend 950PR/Ascend 950DT、Kirin9030后端及CPU仿真（`__CPU_SIM`）上提供C++内建实现，并已登记到虚拟ISA索引（`PTOISA`、`isa/README`、`manifest.yaml`、指令族矩阵 `appendix-d` 及mkdocs导航）。仅Ascend 950PR/Ascend 950DT / Kirin9030 / CPU仿真可用（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品不支持）；尚无公开字节码编码。

## 简介

按源Tile元素的某个**字节**统计直方图（每个字节取值0–255的出现次数），并可按“已处理的高位字节等于给定索引”做级联过滤。它是**基数排序（radix sort）的字节桶计数原语**：第一趟对最高有效字节（MSB）计数；后续各趟对更低字节计数，但仅统计高位字节与上一趟桶索引匹配的元素，从而得到当前基数位在前缀桶内的分布。

一次调用对源Tile的每个有效行独立产出一组256个 `uint32` 计数（bin）。

## 数学语义

设源 `src` 有效形状为 $R \times C$，元素为 `uint16_t` 或 `uint32_t`。记 $B_k(x)$ 为元素 $x$ 的第 $k$ 字节：

$$
B_0 = \text{bits } 7\text{–}0\ (\text{LSB}),\quad B_1 = \text{bits } 15\text{–}8,\quad B_2 = \text{bits } 23\text{–}16,\quad B_3 = \text{bits } 31\text{–}24\ (\text{MSB})
$$

模板参数 `byte` 选择被统计的字节 $k\in\{0,1,2,3\}$。对每一源行 $r\in[0,R)$ 与每一桶值 $b\in[0,256)$：

$$
\mathrm{dst}_{r,b} = \bigl|\{\,j\in[0,C)\ \big|\ B_k(\mathrm{src}_{r,j})=b\ \wedge\ F_{k}(r,j)\ \}\bigr|
$$

其中级联过滤 $F_k$ 按高位优先（先处理 $k=3$，再 $k=2,1,0$）定义：

| 源dtype | `byte` $k$ | 过滤 $F_k(r,j)$ | `idx` 含义 |
|----------|-----------|-----------------|-----------|
| `uint16` | `BYTE_1` (MSB) | 恒真（首趟，不过滤） | 未使用 |
| `uint16` | `BYTE_0` (LSB) | $B_1(\mathrm{src}_{r,j})=\mathrm{idx}_{r}$ | 每行1个匹配字节（高位） |
| `uint32` | `BYTE_3` (MSB) | 恒真（首趟，不过滤） | 未使用 |
| `uint32` | `BYTE_2` | $B_3=\mathrm{idx}_{r,0}$ | 1行过滤字节 |
| `uint32` | `BYTE_1` | $B_3=\mathrm{idx}_{r,0}\ \wedge\ B_2=\mathrm{idx}_{r,1}$ | 2行过滤字节 |
| `uint32` | `BYTE_0` (LSB) | $B_3=\mathrm{idx}_{r,0}\wedge B_2=\mathrm{idx}_{r,1}\wedge B_1=\mathrm{idx}_{r,2}$ | 3行过滤字节 |

- `dst` 每行恒为256个 `uint32` 桶（对应字节取值0–255）。
- `uint16` 源仅支持 `BYTE_0` / `BYTE_1`（只有两个字节可被提取）。
- 除非另有说明，语义在有效区域内定义；桶计数的具体内存交错布局（N0/N1双库、奇偶拆分）为实现定义，逻辑结果为每行256个计数。

> `src`、`idx` 均为ISA可见的Tile操作数（非编译器scratch）。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`，在Ascend 950PR/Ascend 950DT / Kirin9030 / CPU仿真下可用（`PTO_NPU_ARCH_A5 || PTO_NPU_ARCH_KIRIN9030 || __CPU_SIM`）。`HistByte` 枚举定义于 `include/pto/common/type.hpp`。
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
enum class HistByte : uint8_t {
    BYTE_0 = 0, // LSB (bits 7-0)
    BYTE_1 = 1, // bits 15-8
    BYTE_2 = 2, // bits 23-16
    BYTE_3 = 3  // MSB (bits 31-24)
};

template <HistByte byte, typename TileDataDst, typename TileDataSrc, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst &dst, TileDataSrc &src, TileDataIdx &idx, WaitEvents &...events);
```

| 参数 | 方向 | 含义 |
|------|------|------|
| `byte` | 模板 | 被统计的字节（`HistByte::BYTE_0`…`BYTE_3`） |
| `dst` | 输出 | 直方图结果Tile，`uint32_t`，行主序，每行256桶 |
| `src` | 输入 | 源数据Tile，`uint16_t` 或 `uint32_t`，行主序 |
| `idx` | 输入 | 级联过滤索引Tile，`uint8_t`，形状随 `byte` 与源dtype变化（见下） |
| `events...` | 输入 | 等待事件（`WaitEvents`），指令前隐式 `TSYNC` |

## Tile尺寸与数据类型

设源有效形状 $R \times C$：

| Tile | dtype | 有效形状 | 布局 | 说明 |
|------|-------|---------|------|------|
| `dst` | `uint32_t` | $R \times 256$ | RowMajor | 每行256个桶计数 |
| `src` | `uint16_t` 或 `uint32_t` | $R \times C$ | RowMajor | 被统计的数据 |
| `idx`（`uint16` 源） | `uint8_t` | $R \times 1$ | ColMajor（DN） | 每行1个匹配字节（高位） |
| `idx`（`uint32` 源） | `uint8_t` | $(3-k) \times C$ | RowMajor | 每行广播1个过滤字节；$k=3$ 时为0行（不使用） |

> `idx` 的物理行数须按32字节块对齐（`PTO_CEIL(rows · sizeof(uint8_t), 32)`）；`uint16` 模式要求 `idx` 为DN布局（`BLayout::ColMajor` + `SLayout::NoneBox`）且恰好1列。

## 支持的输入dtype

| 源dtype | 目的dtype | idx dtype | 允许的 `byte` | 说明 |
|----------|-----------|-----------|--------------|------|
| `U16` (`uint16_t`) | `U32` | `U8` | `BYTE_0`, `BYTE_1` | 仅有低/高字节可提取 |
| `U32` (`uint32_t`) | `U32` | `U8` | `BYTE_0`…`BYTE_3` | 四字节均可，配合0–3行idx |

> `dst` 必须为 `uint32_t`，`idx` 必须为 `uint8_t`，`src` 限定 `uint16_t` / `uint32_t`；其它组合由实现内 `static_assert` 拦截。

## 实现说明

THISTOGRAM在向量流水线（`PIPE_V`）上执行：

1. **字节提取**：将源元素解交错为按字节向量——`uint16` 走 `DINTLV_B8`（拆出MSB/LSB），`uint32` 走 `DINTLV_B16` + `vdintlv`（拆出4个字节）。
2. **级联过滤**：用 `vcmp_eq` 生成“已处理高位字节 == idx”的谓词，逐字节级联相与（首趟MSB无过滤）。
3. **字节直方图**：以硬件 `chistv2` 在过滤谓词下对选中字节计数，内部以N0/N1双库、奇偶拆分累加；最终每行写回256个 `uint32` 桶（`INTLV_B32` 交错存储）。双库与奇偶拆分属实现细节，逻辑结果为每行256个计数。

## 约束

| 约束 | 适用范围 | 原因 |
|------|---------|------|
| `dst` 为 `uint32_t` 且行主序 | 所有目标 | 256桶计数宽度与存储布局 |
| `src` ∈ {`uint16_t`, `uint32_t`} 且行主序 | 所有目标 | 字节提取路径 |
| `idx` 为 `uint8_t` | 所有目标 | 过滤字节字宽 |
| `uint16` 源：`idx` 为DN（ColMajor + NoneBox）且1列 | Ascend 950PR/Ascend 950DT / Kirin9030 / CPU | 单字节/行广播匹配 |
| `uint32` 源：`idx` 行主序，行数 $=3-k$，列数 $=$ 源列数 | Ascend 950PR/Ascend 950DT / Kirin9030 / CPU | 级联过滤所需索引行 |
| `uint16` 源仅允许 `BYTE_0` / `BYTE_1` | 所有目标 | `uint16` 仅2字节 |
| `dst` 每行256桶 | 所有目标 | 字节取值空间0–255 |

## 示例

```cpp
// uint16 源：对每个元素的高字节（MSB）做直方图（基数排序第一趟）。
THISTOGRAM<HistByte::BYTE_1>(dstTile, srcTile, idxTile);

// uint16 源：对低字节（LSB）做直方图，仅统计高字节 == idx 的元素（第二趟）。
THISTOGRAM<HistByte::BYTE_0>(dstTile, srcTile, idxTile);
```

典型Tile声明（`uint16` 模式，源有效形状 $R\times C$）：

```cpp
using TileDataSrc = Tile<TileType::Vec, uint16_t, R, alignedC,        BLayout::RowMajor>;
using TileDataDst = Tile<TileType::Vec, uint32_t, R, 256,             BLayout::RowMajor>;
using TileDataIdx = Tile<TileType::Vec, uint8_t,  alignedIdxBytes, 1, BLayout::ColMajor>;
```

完整ST示例见 `tests/npu/a5/src/st/testcase/thistogram/`（A5）、`tests/npu/kirin9030/src/st/testcase/thistogram/`（Kirin9030）、`tests/npu/kirinX90/src/st/testcase/thistogram/`（KirinX90）及 `tests/cpu/st/testcase/thistogram/`（CPU参考实现）。
