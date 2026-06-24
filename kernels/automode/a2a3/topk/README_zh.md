# 基础 Topk 算子示例

## 概览

本示例展示如何使用 PTO 实现一个基础 Topk 算子，包含工程结构、构建与执行流程。

## 支持的 AI 处理器

- A2/A3

## 目录结构

```
kernels/topk/
├── scripts/
│   └── gen_data.py              # 生成输入与 golden 输出
├── CMakeLists.txt               # 构建配置
├── topk_kernel.cpp              # Kernel 实现
├── main.cpp                     # Host 侧入口
└── run.sh                       # 便捷脚本
```

## 算子说明

### 计算功能

本示例实现固定维度 `[rows, cols] = [4800, 1024]` 的 Topk：

### 规格

| 项目        | 值 |
| ----------- | ----- |
| OpType      | `topk` |
| 输入        | `[rows, cols] = [4800, 1024]` |
| 输出        | `data`, `index` |
| Kernel 名称 | `topk_kernel` |

### Tiling 参数

验证平台有 48 个核。

每核形状：

- `rows = 100`, `cols = 1024`

## 实现说明

### 类型定义

Topk 实现： 从GM上加载数据和索引到ub上，使用TSort对每32个数据进行排序，使用TMrgsort对每个Tile内部做归并排序. 分别取出topk个数据和索引，存回GM.

```cpp
    // data
    using DynShapeDim5 = Shape<1, 1, 1, singleLoopRow, validCol>;
    using DynStridDim5 = Stride<singleLoopRow * Cols, singleLoopRow * Cols, singleLoopRow * Cols, Cols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    // index
    using IndexShapeDim5 = Shape<1, 1, 1, 1, validCol>;
    using IndexStridDim5 = Stride<validCol, validCol, validCol, validCol, 1>;
    using IndexGlobalData = GlobalTensor<indexT, IndexShapeDim5, IndexStridDim5>;

    // sorted data and index
    using DstShapeDim5 = Shape<1, 1, 1, singleLoopRow, topk>;
    using DstStridDim5 = Stride<singleLoopRow * topk, singleLoopRow * topk, singleLoopRow * topk, topk, 1>;
    using DstDataGlobalData = GlobalTensor<T, DstShapeDim5, DstStridDim5>;
    using DstIdxGlobalData = GlobalTensor<indexT, DstShapeDim5, DstStridDim5>;
```

### 流水线调度

本示例通过在 UB 上使用双缓冲来重叠数据搬运与计算，以提高利用率。每次循环执行两组操作，TLOAD->TSORT32->TMRGSORT(含MRGSORT和MOV)->TSTORE。单组操作的依赖顺序是MTE2->V->MTE1->V->MTE3。第二组的TLOAD不需要等第一组操作全部执行完再开始执行，这样增加了流水并行度。增加了循环之间的从V->MTE2的反向依赖，以保证下一个循环的TLOAD是在对应的VEC操作执行完后再开始的。

## 实测性能（参考）

以下数据在 Ascend A3（48个VEC核）上测得，覆盖多个尺寸以及不同类型。

| 参数 | aiv_vec_ratio | aiv_scalar_ratio | aiv_mte2_ratio | aiv_mte3_ratio | task_duration(us) |
| --- | --- | --- | --- | --- | --- |
| `type=float` `validRow=rows=4800` `validCol=1024` `cols=1280` `topk=1000` | 94% | 3.2% | 11.7% | 10.4% | 324.106 |
| `type=float` `validRow=rows=3456` `validCol=1024` `cols=1280` `topk=1000` | 91.5% | 4.6% | 12.3% | 10.5% | 238.819 |
| `type=float` `validRow=rows=2304` `validCol=1024` `cols=1280` `topk=1000` | 88.7% | 6% | 12.4% | 10.1% | 161.375 |
| `type=half` `validRow=rows=4800` `validCol=1024` `cols=1280` `topk=1008` | 93.7% | 2.4% | 11.5% | 9.6% | 326.886 |

## 构建与运行

1. 配置 Ascend CANN 环境（示例路径）：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 运行示例：

```bash
cd ${git_clone_path}/kernels/manual/a2a3/topk
bash run.sh -r npu -v Ascend910B1
```

成功时输出：

```text
test success
```
