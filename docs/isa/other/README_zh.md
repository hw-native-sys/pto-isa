# 其他与通信

本节包含不属于 Tile、Vector 或标量/控制主干的残余指令和通信操作。

## 两大分类

### 通信与运行时（Communication and Runtime）

核间集体通信与同步原语。

| 指令 | 说明 | 同步类型 |
|------|------|----------|
| [TBROADCAST](../comm/TBROADCAST_zh.md) | 从 root NPU 广播数据到所有 rank | 同步 |
| [TGET](../comm/TGET_zh.md) | 从远程 NPU 获取数据 | 同步 |
| [TGET_ASYNC](../comm/TGET_ASYNC_zh.md) | 从远程 NPU 异步获取数据 | 异步 |
| [TPUT](../comm/TPUT_zh.md) | 向远程 NPU 发送数据 | 同步 |
| [TPUT_ASYNC](../comm/TPUT_ASYNC_zh.md) | 向远程 NPU 异步发送数据 | 异步 |
| [TNOTIFY](../comm/TNOTIFY_zh.md) | 通知其他 rank 某个事件发生 | 同步 |
| [TWAIT](../comm/TWAIT_zh.md) | 等待通知到达 | 同步 |
| [TTEST](../comm/TTEST_zh.md) | 测试通知是否已到达 | 同步 |
| [TGATHER](../comm/TGATHER_zh.md) | 从所有 rank 收集数据到 root NPU | 同步 |
| [TSCATTER](../comm/TSCATTER_zh.md) | 从 root NPU 散发数据到所有 rank | 同步 |
| [TREDUCE](../comm/TREDUCE_zh.md) | 在所有 rank 上做集体归约 | 同步 |

[通信与运行时契约 →](./communication-and-runtime_zh.md)

### 非 ISA 支撑操作（Non-ISA Supporting Operations）

面向 tile 序列或内存管理的便利操作。

| 操作 | 说明 | 分类 |
|------|------|------|
| [TALIAS](../TALIAS_zh.md) | 创建 tile 的别名视图（无数据拷贝） | 别名 |
| [TAXPY](../TAXPY_zh.md) | 融合乘加：`dst = src0 * scalar + src1` | 融合计算 |
| [TCONCAT](../TCONCAT_zh.md) | 沿维度拼接两个 tile | tile 序列 |
| [TDEQUANT](../TDEQUANT_zh.md) | 从量化格式反量化 tile | 量化 |
| [TFREE](../TFREE_zh.md) | 释放先前分配的 tile 或 buffer | 内存 |
| [THISTOGRAM](../THISTOGRAM_zh.md) | 计算 tile 值的直方图 | 统计 |
| [TPACK](../TPACK_zh.md) | 将多个 tile 打包进单个 tile buffer | tile 序列 |
| [TPOP](../TPOP_zh.md) | 谓词 mask 的置 1 位计数 | 谓词 |
| [TPUSH](../TPUSH_zh.md) | 谓词 mask 的置 0 位计数 | 谓词 |
| [TRANDOM](../TRANDOM_zh.md) | 用随机值填充 tile | 生成 |
| [TQUANT](../TQUANT_zh.md) | 将 tile 量化为整数格式 | 量化 |

[非 ISA 支撑操作契约 →](./non-isa-and-supporting-ops_zh.md)

## 相关页面

| 页面 | 内容 |
|------|------|
| [指令集总览](../instruction-surfaces/other-instructions_zh.md) | 其他指令集的高层描述 |
| [指令族](../instruction-families/README_zh.md) | 所有指令集的规范契约 |
| [指令集总览](../instruction-surfaces/README_zh.md) | 四大指令集地图 |
