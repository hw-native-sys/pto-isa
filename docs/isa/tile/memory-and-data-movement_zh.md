# 内存与数据搬运指令集

这一组指令负责在 GM 与 tile 本地缓冲之间搬运数据，也是 tile 路径里唯一直接跨越“GM 可见状态”和“tile 可见状态”的指令面。无论是连续搬运，还是带索引的 gather/scatter，都属于这一类。

## 指令一览

| 操作 | 说明 |
| --- | --- |
| `pto.tload` | 从 GlobalTensor 加载到 tile |
| `pto.tprefetch` | 预取数据到本地缓冲 |
| `pto.tstore` | 从 tile 写回 GlobalTensor |
| `pto.tstore_fp` | 经 fix-pipe 路径写回 |
| `pto.mgather` | 索引式 gather |
| `pto.mscatter` | 索引式 scatter |

## 机制

### 连续搬运：TLOAD / TSTORE

二维视角下可以理解为：

```text
TLOAD:  dst[i, j] = src[r0 + i, c0 + j]
TSTORE: dst[r0 + i, c0 + j] = src[i, j]
```

其中真正的搬运尺寸不是物理 tile 尺寸，而是 valid region：

- `TLOAD` 由目标 tile 的 valid region 决定搬运范围
- `TSTORE` 由源 tile 的 valid region 决定写回范围

### 预取：TPREFETCH

`TPREFETCH` 会提前把后续可能要用到的 GM 数据搬进 tile 可见的本地路径。它的意义不在于改变数据布局，而在于把“稍后会访问的数据”尽早拉近。

### 索引搬运：MGATHER / MSCATTER

索引类搬运允许非连续地址访问：

```text
gather:  dst[i] = src[index[i]]
scatter: dst[index[i]] = src[i]
```

### `_fp` 变体

`TSTORE_FP` 里的 `_fp` 指的是 **fix-pipe** 路径，不是 floating-point store 的缩写。它通过 fix-pipe sideband state 配合写回。

## 布局兼容性

| TileType | ND→ND | DN→DN | NZ→NZ | ND→NZ | DN→ZN | 说明 |
| --- | :---: | :---: | :---: | :---: | :---: | --- |
| `TileType::Vec` | Yes | Yes | Yes | No | No | |
| `TileType::Mat` | Yes | Yes | Yes | Yes | Yes | |
| `TileType::Acc` | Yes | No | Yes | No | No | 原子写回等特殊路径 |

### A5 额外限制

- `TileType::Vec` 若做 `ND→NZ` 或 `DN→ZN`，要求 `GlobalData::staticShape[0..2] == 1` 且 `TileData::SFractalSize == 512`
- `int64_t/uint64_t` 的 Vec 路径只支持 `ND→ND` 或 `DN→DN`

## 目标 Profile 支持

| 元素类型 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `f32` / `f16` / `bf16` | Yes | Yes | Yes |
| `i8/u8` | Yes | Yes | Yes |
| `i16/u16` | Yes | Yes | Yes |
| `i32/u32` | Yes | Yes | Yes |
| `i64/u64` | Yes | Yes | Yes |
| `f8e4m3 / f8e5m2` | No | No | Yes |
| `hifloat8_t / float4_e*` | No | No | Yes |

## 排序与同步

内存与数据搬运指令必须遵守 PTO 的生产者-消费者排序规则。程序不能假设“搬完了自然可见”，而是应显式使用：

- `TSYNC`
- `set_flag / wait_flag`

来保证后续计算看到的是已经到位的数据。

## 约束

- 源与目标 dtype 的字节大小必须兼容：`sizeof(tile.dtype) == sizeof(gtensor.dtype)`
- 布局兼容性依赖 target profile
- gather/scatter 的索引 tile 必须满足形状和类型约束
- `TSTORE` 的 `AtomicType` 是否可用由 profile 决定
- `TSTORE_FP` 当前只对 `TileType::Acc` 的 fix-pipe 路径合法

## 不允许的情形

- 使用未初始化的 tile 参与搬运
- GlobalTensor 的 stride 与搬运模式不兼容
- 访问超出张量声明 shape 的 GM 地址
- 对非 `TileType::Acc` 使用 `TSTORE_FP`
- 在 CPU 模拟器上假设原子写回具备 NPU 上的所有语义

## 相关页面

- [一致性基线](../memory-model/consistency-baseline_zh.md)
- [生产者-消费者排序](../memory-model/producer-consumer-ordering_zh.md)
- [Tile 指令族](../instruction-families/tile-families_zh.md)
