# A6R2 MaxTrack C2V（本 ST 与最新 TPUSH/TPOP 接口）

MaxTrack 挂在 Acc→UB fixpipe（`TMovCcToUb` + `set_fixp_max_cfg`）上：Cube `TPUSH` 写出 vec，并随路写 GroupMax / RowMax 旁路；Vector 用 MaxTrack `TPOP` 一并弹出。

实现位置：`include/pto/npu/a6r2/TPush.hpp`、`TPop.hpp`、`TMov.hpp`；配置在 `include/pto/common/fixpipe.hpp`。

---

## 1. 配置：`MaxTrack` + `FixpipeParams`

```cpp
template <bool absEn = false, bool groupmaxEn = false, bool rowmaxEn = false,
          bool rowmaxCompareEn = false, bool rowmaxOutput = false, uint8_t groupNum = 8>
struct MaxTrack { ... };

// 嵌入 FixpipeParams 最后一参
using Cfg = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::NoQuant, ..., MaxTrackCfg>;
TPUSH<Pipe, AccTile, Cfg>(pipe, acc);
```

| `MaxTrack` 字段 | 含义 |
|-----------------|------|
| `abs_en` | 求 max 前是否取绝对值 |
| `groupmax_en` | 输出 GroupMax；为 true 时走 MaxTrack C2V（否则同普通 Acc→UB） |
| `rowmax_en` | 输出 RowMax（需同时 `groupmax_en`） |
| `rowmax_compare_en` | 是否与硬件 **MAX_BUFF** 比较（见下） |
| `rowmax_output` | 是否把 RowMax 写到 UB |
| `group_num` | 每组元素数（默认 8） |

输出组合：

- `groupmax_en` → `TPOP(pipe, vec, gMax)`
- `groupmax_en && rowmax_en` → `TPOP(pipe, vec, gMax, rMax)`

dtype：float / half / bf16。布局仅 **NZ2ND / NZ2DN** 时 GroupMax/RowMax 生效。

---

## 2. `TPipe` 构造（绑定旁路 UB）

```cpp
TPipe<FlagID, DIR_C2V, SlotBytes, SlotNum, LocalSlotNum, /*IsNoSplit*/true>
    pipe(gm, C2V_BUF, v2cBuf, C2V_GROUPMAX_BUF, C2V_ROWMAX_BUF);
```

- `groupmax_addr` / `rowmax_addr` 由 pipe 的 `C2V_GROUPMAX_BUF` / `C2V_ROWMAX_BUF` 换算（**C0=32B 单位**），不写在 `MaxTrack` 里。
- MaxTrack C2V 推送使用 `TILE_NO_SPLIT`。

---

## 3. TPUSH 接口行为

```cpp
TPUSH<Pipe, AccTile, FixpipeParams<..., MaxTrack<...>>>(pipe, acc);
```

### 旁路形状

| 旁路 | NZ2ND | NZ2DN |
|------|-------|-------|
| GroupMax | `[validRow, validCol/group_num]` | `[validCol/group_num, validRow]` |
| RowMax | `[1, validRow]`（存储 Cols 按 32B 上取整） | 同左 |

`vec` / `gMax` UB 仍按 `tileIndex % SLOT_NUM` 轮转。

### `rowmax_compare_en` 与 RowMax 槽

| `rowmax_compare_en` | 行为 |
|---------------------|------|
| `false`（seed） | 不与 MAX_BUFF 历史比；占用**新** m 条带 RowMax 槽：`rowmaxProd >= SLOT_NUM` 时 `WaitRowMaxSlot`（`FlagID+2`），再 `slot = rowmaxProd % SLOT_NUM`，`rowmaxProd++` |
| `true`（cmp） | 与 MAX_BUFF 比较并回写；**复用**当前条带槽 `(rowmaxProd - 1) % SLOT_NUM`（要求此前已 seed） |

本 ST：每个 `mLoop` 内 `nTile==0` 用 seed 配置，其余 `nTile` 用 cmp 配置 → 一条 M 上跨 N 的最终行 max 在 HW 侧完成，**无需软件 TMAX**。

---

## 4. TPOP 接口行为

```cpp
// 仅 GroupMax
TPOP<Pipe, Vec, GMax, TConfig, Split>(pipe, vec, gMax);

// GroupMax + RowMax；EndRowMaxCmp 默认 false；TConfig 提供 group_num
TPOP<Pipe, Vec, GMax, RMax, TConfig, Split, EndRowMaxCmp>(pipe, vec, gMax, rMax);
```

- `TPOP` 按 `popTile` 的 valid + 布局设置 `gMax` / `rMax` 的 valid shape。
- **RowMax 地址**：`C2V_ROWMAX_BUF + (fifo.rowmaxCons % SLOT_NUM) * rMaxSlotSize`（按 m 条带，**不**跟 `tileIndex`）。
- **`EndRowMaxCmp=true`**（本 ST：每个 `mLoop` 最后一个 `nTile`）：
  - `rowmaxCons++`
  - 若 `rowmaxCons >= SLOT_NUM`，`NotifyRowMaxSlot`（`FlagID+2`），允许 Cube 复用同号槽
    （必须用 `>=` 而非 `>`：否则第 `SLOT_NUM` 次 seed 的 `Wait` 会等当前条带自己的 `End`，与 `TPOP` 死锁）
  - 本次已 `TASSIGN` 的 `rMaxTile` 地址不变，随后仍可 `TADD`
  - `>=` 会比 seed `Wait` 多 1 次 notify；Cube 在循环结束后 `WaitRowMaxSlot` 一次 drain

---

## 5. RowMax m 级槽与 credit（小结）

```
slot = index % SLOT_NUM
addr = C2V_ROWMAX_BUF + slot * rMaxSlotSize
```

| 侧 | 计数 | 动作 |
|----|------|------|
| Cube | `fifo.rowmaxProd` | seed 换槽 + 必要时 wait；cmp 复用上一槽 |
| Vector | `fifo.rowmaxCons` | 整段 m 的 TPOP 读同一槽；`EndRowMaxCmp` 时递增并可能 notify |

- **同一 `mLoop`（nLoop 内）**：RowMax UB 地址不变。
- **换 `mLoop`**：换槽；Cube/Vector 用 `FlagID+2` 做 m 级占用/释放（与 tile free 的 `FlagID+1` 分离）。
- 计数在各自 core 的 `RingFIFO` 上；跨核只靠 flag。

---

## 6. 本 ST 用法对照

### `case_maxtrack_512_128_128_512`（m×n）

形状：`A[512,128] @ B[128,512] → C[512,512]`，tile 128，`mLoops=nLoops=4`，`group_num=8`。

| 步骤 | 接口 |
|------|------|
| Cube | `nTile==0`：`TPUSH<..., FixpipeConfigInit>`（`compare_en=false`）；否则 `FixpipeConfigCmp`（`true`） |
| Vector | 非末 n：`TPOP<..., false>`；末 n：`TPOP<..., true>`（`EndRowMaxCmp`） |
| 写出 | vec/gMax：每 tile `TSTORE` 拼满 `[512,512]` / `[512,64]`；rMax：每 m 一次 `TSTORE` → `[512,1]` |
| Golden | 全图 GroupMax / RowMax（`abs_en` 先取绝对值），再 `+2`；vec 为原始 C+2 |

### `case_maxtrack_512_128_128_128`（仅 mLoop）

形状：`A[512,128] @ B[128,128] → C[512,128]`，tile 128，`mLoops=4`，`nLoops=1`，`group_num=8`。

| 步骤 | 接口 |
|------|------|
| Cube | 仅 seed：`TPUSH<..., FixpipeConfig>`（`compare_en=false`，无跨 N cmp） |
| Vector | 每次：`TPOP<..., EndRowMaxCmp=true>` |
| 写出 | vec `[512,128]`，gMax `[512,16]`，rMax `[512,1]`（每 m 直接 `TSTORE`） |
| Golden | 同上（`abs_en` 先 abs 再 max；vec 为原始 C+2） |

---

## 7. 硬件约束

- NZ2ND + `groupmax_en`：`N / group_num ≤ 16`
- `rowmax_en`，或 NZ2DN + `groupmax_en`：`validRow ≤ 128`
- GroupMax / RowMax UB 地址 32B 对齐
- `FlagID+2 ≤ MAX_SYC_ID`（RowMax credit）
