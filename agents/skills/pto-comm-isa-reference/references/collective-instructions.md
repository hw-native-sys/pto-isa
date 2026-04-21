# 集合通信指令详解（TGATHER / TSCATTER / TBROADCAST / TREDUCE）

所有集合通信指令共享以下特征：
- 仅 **root** 执行调用，非 root 不得调用（未定义行为）
- 基于 `ParallelGroup` 指定参与者
- 支持单缓冲和乒乓双缓冲
- 数据超出 UB Tile 时自动二维滑动分块

---

## TGATHER — 多 rank 收集

Root 从所有 rank 收集数据，沿 DIM_3 拼接。

```cpp
// 单缓冲
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
RecordEvent TGATHER(ParallelGroupType &group, GlobalDstData &dst,
                    TileData &stagingTile, WaitEvents&... events);

// 乒乓
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
RecordEvent TGATHER(ParallelGroupType &group, GlobalDstData &dst,
                    TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

### 约束

- `dstGlobalData` 指向本地内存，`GetShape(DIM_3)` 必须 ≥ `N × H`
- `parallelGroup.tensors[r]` 指向 rank r 的远端源缓冲区
- 所有源 tensor 必须形状和步幅相同
- Tile 分块约束：静态 `ValidRow`/`ValidCol` 必须能整除对应维度

### 示例

```cpp
GPerRank tensors[NRANKS];
for (int i = 0; i < NRANKS; ++i) tensors[i] = GPerRank(group_addrs[i]);

comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
GResult dstG(result);
TileT stagingTile(TILE_ROWS, TILE_COLS);
comm::TGATHER(group, dstG, stagingTile);
```

---

## TSCATTER — 从 root 分发

Root 将数据沿 DIM_3 拆分后分发到各 rank。TGATHER 的逆操作。

```cpp
// 单缓冲
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
RecordEvent TSCATTER(ParallelGroupType &group, GlobalSrcData &src,
                     TileData &stagingTile, WaitEvents&... events);

// 乒乓
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
RecordEvent TSCATTER(ParallelGroupType &group, GlobalSrcData &src,
                     TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

### 约束

- `srcGlobalData` 指向本地内存，`GetShape(DIM_3)` 必须 ≥ `N × H`
- `parallelGroup.tensors[r]` 指向 rank r 的远端目标缓冲区

---

## TBROADCAST — 广播

Root 将本地数据广播到所有 rank。

```cpp
// 单缓冲
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
RecordEvent TBROADCAST(ParallelGroupType &group, GlobalSrcData &src,
                       TileData &stagingTile, WaitEvents&... events);

// 乒乓
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
RecordEvent TBROADCAST(ParallelGroupType &group, GlobalSrcData &src,
                       TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

### 约束

- `srcGlobalData` 指向本地内存
- `parallelGroup.tensors[k]` 指向 rank k 的远端目标缓冲区

---

## TREDUCE — 多 rank 归约

Root 从所有 rank 收集数据并执行逐元素归约。

```cpp
// 基础 reduce（累加 Tile + 接收 Tile）
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
RecordEvent TREDUCE(ParallelGroupType &group, GlobalDstData &dst,
                    TileData &accTile, TileData &recvTile, ReduceOp op, WaitEvents&... events);

// 乒乓 reduce
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
RecordEvent TREDUCE(ParallelGroupType &group, GlobalDstData &dst,
                    TileData &accTile, TileData &pingTile, TileData &pongTile,
                    ReduceOp op, WaitEvents&... events);
```

### 约束

- `dstGlobalData` 指向本地内存
- `accTileData`、`recvTileData`（或 `accTile` + `pingTile` + `pongTile`）必须为预先分配的 UB Tile
- `parallelGroup.tensors[r]` 指向 rank r 的远端源缓冲区
- 分块约束同 TGATHER

### 示例

```cpp
comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
GTensor dstG(result);
TileT accTile, recvTile;
comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
```
