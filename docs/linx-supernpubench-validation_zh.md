# SuperNPUBench one-level-arch 验证

[English](linx-supernpubench-validation.md)

验证范围为 [cases.json](../tests/cpu/linx_supernpubench/cases.json) 中的 544 条记录：444 条活跃构建配置、75 条补充源码配置、25 条追加数值配置。这是配置数，不是 544 个独立算子。其中 78 条有独立数值检查，其余仅验证执行完成。

CPU 共通过 **462/544** 条，覆盖原生全部 **428/428** 条成功记录，没有原生通过而 CPU 失败的项。其中原生 **58** 条数值通过在 CPU 也通过独立 golden。

| 结果 | CPU-SIM | Linx 原生 |
| --- | ---: | ---: |
| 数值通过 | 71 | 58 |
| 仅执行完成 | 391 | 370 |
| 构建/API 失败 | 76 | 60 |
| 编译器失败 | 0 | 30 |
| 执行失败 | 1 | 16 |
| 执行超时 | 1 | 0 |
| 数值失败 | 4 | 10 |

## 复现与证据

SuperNPUBench 固定为 `f6359b207dac65eca7332b4d527d6ccfca95f722`；`d34cea27e3d03228c2d897f83417ce1e7cde84e1` 的 benchmark 源码相同。原生 API 为 `74d7326c521aeb1b21c52602b67c55b42b3563a3`，模型 checkout 为 `34c110137a5c039f0283118749450e7966556118`，使用 Linx Clang 15.0.4 和匹配的 musl，主机为 aarch64/GCC 15.2.1。编译关闭 fast-math 和浮点融合，按每条记录启动 1 或 4 个 PE。

安装 NumPy/PyTorch 后运行：

```bash
python3 tests/cpu/test_linx_supernpubench.py \
  --benchmark /path/to/SuperNPUBench \
  --upstream /path/to/Linx-TileOP-API \
  --toolchain /path/to/linx_blockisa_llvm_musl \
  --model /path/to/SuperScalarModel \
  --musl-libs /path/to/musl-only-libs \
  --compiler /path/to/g++-15 --routes cpu,upstream \
  --jobs 16 --output-dir /tmp/linx-supernpubench
```

输出目录必须为空。`--ids S0063,C0408` 选择记录；`--routes cpu` 只需要样例仓和主机编译器；`--sanitize` 启用 ASan/UBSan。每次编译/执行默认限时 240 秒，可用 `--timeout` 调整。两路同时运行时，原生通过而 CPU 未通过会使 runner 返回非零；两路均失败的已知配置仍完整保留，退出零不代表 544 条全部通过。

`results.json` 保存完整命令和分类，`provenance.json` 保存源码、后端头文件、adapter 和模型哈希。单例证据位于 `results/<ID>/<route>/`。数值输入只生成一次并原样复制到两路，记录 SHA-256；golden 使用上游独立 NumPy/PyTorch 实现及容差。模型必须输出 guest `R2 = 0` 且没有错误/断言，进程退出零本身不算通过。

## CPU 适配范围

原生路线使用未经修改的样例源码；CPU 路线使用隔离副本和显式 [host adapter](../tests/cpu/linx_supernpubench/host_adapter.py)，完整差异保存在 `host/SuperNPUBench/benchmark/host-adapter.patch`。因此结果表示经过所列移植适配的 CPU 验证。

- 启动器在编译后重命名 `main` 符号，保留 C++ 隐式返回语义，并以共享 Core 上下文启动 PE；RES_CHECK 标志使用主机原子变量。
- GNU C 区间初始化转换为等值的 main 前初始化；驱动内的 `sync` 名称避开 POSIX 同名函数。
- 哈希、histogram/top-k、QLI 的原生汇编 emitter 映射为 CPU Tile 操作；仅删除 QLI trace hint。动态卷积的空 Shared 寄存器输出约束映射为显式 slot 发布/获取。
- MoE combine 动态驱动在 reset/input 初始化后增加屏障，避免其他 PE 的 reset 覆盖 PE0 更新。
- 启动器显式配置零初始化 GM 区域 `[0, 1 MiB)` 和 `[0x4000800000, 0x4001800000)`，QLI 标量物理指针显式转换。这些是模型内存假设，不能将动态 QLI 的运行通过视为外部数据集数值通过。缺少嵌入数据对象的 QLI/sparse-MLA 配置仍报告构建失败。

没有将内核替换为主机参考算法，也没有修改 golden。上述 emitter 和驱动适配属于验证设置，应与后端本身的数值/runtime 实现一起审阅。

## 后端行为与失败分析

覆盖 BF16 标量舍入、物理容量推导、动态 reduction-prefix 视图、RowMajor/CUBE TileArray、GPR 掩码、允许别名的输入快照，以及多 PE 屏障、异常传播、显式 GM 映射和 Shared slot 的有序发布/消费。每个 PE 的 slot 消费位置独立，慢消费者不会被快生产者覆盖；复制 handle 可复用已取得的 generation。这不表示模拟了所有非法别名、全芯片容量、内存域或有界硬件 FIFO 背压；原生 API 没有 TPUSH/TPOP。接口用法见[后端契约](../include/pto/npu/linx/README_zh.md)。

[逐项机器可读矩阵](../tests/cpu/linx_supernpubench/validation-20261112.json) 保存所有状态及原生首条相关诊断；ID 对应 manifest 中的目录和完整参数。剩余失败分别涉及样例契约、编译器描述符、模型执行和数值检查，不能统一归因于 CPU 后端。

在独立诊断副本中，仅将 packed GM 两处描述符列数从 `kStoredGK` 改为逻辑 `gK`，保持输入和 golden 后，P0008/P0009 在两路均数值通过；矩阵仍记录原始源码的失败。

已定位四位 packed GM 行跨度重复除以二、按行索引调用字节偏移 MGATHER、MoE reset 竞争和 descriptor reinterpret-cast，以及缺少嵌入输入对象。额外开启哈希驱动的自检后，两路均在 139 条查询/4 PE 条件下漏算 11 条尾部结果；执行完成不等于数值正确。

Fixp 量化样例将 FP19 scale 打包到 `[31:13]`，本次模型却解码低 19 位，解释了原生输出接近零或仅剩 offset 的现象；CPU 遵循 packed descriptor 契约。Conv2d 使用 ColMajor GM 配合默认 CUBE load/store selector，需要核查布局约定。其他 compiler/model 描述符失败仍需对应组件定位。
