# 其他与通信

本节覆盖那些不适合归入 Tile、Vector 或标量/控制主干的架构可见操作。

## 通信与运行时

跨 NPU 的点对点与集合通信。

| 指令 | 说明 |
|------|------|
| [TBROADCAST](../comm/TBROADCAST_zh.md) | 从根 NPU 向所有 rank 广播数据 |
| [TGET](../comm/TGET_zh.md) | 从远端 NPU 读取数据 |
| [TGET_ASYNC](../comm/TGET_ASYNC_zh.md) | `TGET` 的异步形式 |
| [TNOTIFY](../comm/TNOTIFY_zh.md) | 通知其他 rank 发生某个事件 |
| [TPUT](../comm/TPUT_zh.md) | 向远端 NPU 写入数据 |
| [TPUT_ASYNC](../comm/TPUT_ASYNC_zh.md) | `TPUT` 的异步形式 |
| [TREDUCE](../comm/TREDUCE_zh.md) | 并行组上的集合归约 |
| [TSCATTER](../comm/TSCATTER_zh.md) | 从根 NPU 向所有 rank 分发数据 |
| [TGATHER](../comm/TGATHER_zh.md) | 从所有 rank 收集数据到根 NPU |
| [TTEST](../comm/TTEST_zh.md) | 测试通知是否已到达 |
| [TWAIT](../comm/TWAIT_zh.md) | 等待通知或信号成立 |

详见 [通信与运行时](./communication-and-runtime_zh.md)。

## 非 ISA 与支撑操作

更高层的 tile 序列、量化和资源管理语义。

| 操作 | 说明 | 类别 |
|------|------|------|
| [TALIAS](../TALIAS_zh.md) | 在不复制数据的前提下创建 tile 别名视图 | Alias |
| [TAXPY](../TAXPY_zh.md) | 融合乘加：`dst = src0 * scalar + src1` | Fused compute |
| [TCONCAT](../TCONCAT_zh.md) | 按指定维度拼接 tile | Tile sequence |
| [pto.tdequant](./ops/non-isa-and-supporting-ops/tdequant_zh.md) | 把量化表示恢复成数值表示 | Quantize |
| [TFREE](../TFREE_zh.md) | 释放先前分配的 tile 或缓冲区 | Memory |
| [THISTOGRAM](../THISTOGRAM_zh.md) | 统计 tile 元素直方图 | Statistics |
| [TPACK](../TPACK_zh.md) | 将多个 tile 打包进单一 tile buffer | Tile sequence |
| [TPOP](../TPOP_zh.md) | 计算谓词 mask 的 population count | Predicate |
| [TPUSH](../TPUSH_zh.md) | 计算谓词 mask 的 push count | Predicate |
| [TRANDOM](../TRANDOM_zh.md) | 用随机值填充 tile | Generation |

详见 [非 ISA 与支撑操作](./non-isa-and-supporting-ops_zh.md)。

## 相关页面

- [其他指令集](../instruction-surfaces/other-instructions_zh.md)
- [其他指令族](../instruction-families/other-families_zh.md)
