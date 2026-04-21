# 数据搬运优化

## 乒乓双缓冲

### 原理

使用两个 UB Tile 交替工作，将 TLOAD（MTE2）和 TSTORE（MTE3）重叠：

```
时间 →
MTE2:  [TLOAD ping] [TLOAD pong] [TLOAD ping] ...
MTE3:            [TSTORE ping] [TSTORE pong] ...
       ↑ 无重叠  ↑───── 重叠区 ──────↑
```

### 实现模式

```cpp
TileData pingTile(ROWS, COLS);
TileData pongTile(ROWS, COLS);
TASSIGN(pingTile, 0x0);
TASSIGN(pongTile, TILE_UB_BYTES);

// 方式 1：使用内置 ping-pong（推荐）
comm::TPUT(dstG, srcG, pingTile, pongTile);
comm::TGET(dstG, srcG, pingTile, pongTile);

// 方式 2：手动 ping-pong（更灵活）
for (int i = 0; i < num_chunks; i++) {
    bool use_ping = (i % 2 == 0);
    TileData &curTile = use_ping ? pingTile : pongTile;
    event_t curEv = use_ping ? EVENT_ID0 : EVENT_ID1;

    if (i > 0) {
        TileData &prevTile = use_ping ? pongTile : pingTile;
        event_t prevEv = use_ping ? EVENT_ID1 : EVENT_ID0;
        wait_flag(PIPE_MTE2, PIPE_MTE3, prevEv);
        TSTORE_IMPL<...>(prevDst, prevTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, prevEv);
        wait_flag(PIPE_MTE3, PIPE_MTE2, prevEv);
    }

    TLOAD(curTile, srcG_i);
    set_flag(PIPE_MTE2, PIPE_MTE3, curEv);
}
// 刷新最后一个 tile
```

### 何时使用乒乓

| 场景 | 建议 |
|------|------|
| 大量小块传输（多次 TLOAD/TSTORE） | 强烈推荐 |
| 单次大块传输 | 不需要（内置指令已自动分块） |
| UB 空间紧张 | 使用单缓冲 |
| 传输量 > 2 × Tile 大小 | 推荐 |

### 内置 vs 手动乒乓

- **内置**（TPUT/TGET 的 ping-pong 重载）：简单场景，自动处理流水线同步
- **手动**：需要在 TLOAD/TSTORE 之间插入自定义逻辑（如 AtomicAdd 选择）

---

## Tile 大小选择

| 考虑因素 | 影响 |
|---------|------|
| UB 容量 | Tile 不能超过 UB 大小（典型 ~192KB） |
| 传输效率 | 大 Tile：更少的传输次数，更高效率 |
| 重叠粒度 | 小 Tile：更早开始通信 |
| 对齐 | 32B 对齐（行方向） |
| 乒乓 | 需要 2× Tile 空间 |

**推荐基线**（half 类型）：

```
UB 约 192KB
乒乓模式需要 2 × Tile
单 Tile ≤ 96KB

128 × 256 × 2B = 64KB  → 安全，乒乓后 128KB
64 × 512 × 2B  = 64KB  → 安全
256 × 256 × 2B = 128KB → 单缓冲可用，乒乓危险
```

---

## 数据对齐

```cpp
// Tile 列数需要 32B 对齐
constexpr int alignedCols = ((cols * sizeof(T) + 31) / 32) * (32 / sizeof(T));

// Tile 间间隔向上对齐到 1024B
constexpr size_t TILE_UB_BYTES = ((M * N * sizeof(T) + 1023) / 1024) * 1024;
```

---

## GM 数据布局

通信数据在 GM 上的布局影响传输效率：

- **连续布局**：最佳，TPUT/TGET 一次传输完成
- **带步长布局**：自动分块按行传输，有额外开销
- **异步传输**：必须一维连续（TPUT_ASYNC/TGET_ASYNC 的约束）
