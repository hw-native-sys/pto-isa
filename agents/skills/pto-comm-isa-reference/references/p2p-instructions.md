# P2P 指令详解（TPUT / TGET）

## TPUT — 远程写

**数据流**：`srcGlobalData（本地 GM）` → `stagingTileData（UB）` → `dstGlobalData（远端 GM）`

当 GlobalTensor 超出 UB Tile 容量时，自动执行二维滑动分块。

### 接口签名

```cpp
// 单 Tile（自动分块）— 编译期原子类型
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
RecordEvent TPUT(GlobalDstData &dst, GlobalSrcData &src, TileData &stagingTile, WaitEvents&... events);

// 单 Tile — 运行时原子类型
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
RecordEvent TPUT(GlobalDstData &dst, GlobalSrcData &src, TileData &stagingTile,
                 AtomicType atomicType, WaitEvents&... events);

// 乒乓双缓冲
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
RecordEvent TPUT(GlobalDstData &dst, GlobalSrcData &src,
                 TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

### 约束

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `TileData::DType == GlobalSrcData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- `dstGlobalData` 必须指向远端地址
- `srcGlobalData` 必须指向本地地址
- `stagingTileData` 必须预先在 UB 中分配
- 乒乓模式：`pingTile` 和 `pongTile` 必须相同类型和维度，不重叠的 UB 偏移
- `atomicType` 支持 `AtomicNone` 和 `AtomicAdd`

### 示例

```cpp
// 基础远程写
comm::TPUT(dstG, srcG, stagingTile);

// 带原子加的远程写
comm::TPUT<AtomicType::AtomicAdd>(dstG, srcG, stagingTile);

// 乒乓双缓冲（自动分块）
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);
comm::TPUT(dstG, srcG, pingTile, pongTile);

// 运行时选择原子类型
comm::TPUT(dstG, srcG, stagingTile, AtomicType::AtomicAdd);
```

---

## TGET — 远程读

**数据流**：`srcGlobalData（远端 GM）` → `stagingTileData（UB）` → `dstGlobalData（本地 GM）`

### 接口签名

```cpp
// 单 Tile
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
RecordEvent TGET(GlobalDstData &dst, GlobalSrcData &src, TileData &stagingTile, WaitEvents&... events);

// 乒乓双缓冲
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
RecordEvent TGET(GlobalDstData &dst, GlobalSrcData &src,
                 TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

### 约束

- 与 TPUT 类似，但方向相反
- `srcGlobalData` 指向远端，`dstGlobalData` 指向本地
- TGET 不支持原子操作

### 示例

```cpp
// 基础远程读
comm::TGET(dstG, srcG, stagingTile);

// 乒乓双缓冲远程读
comm::TGET(dstG, srcG, pingTile, pongTile);
```
