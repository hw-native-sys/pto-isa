# Linx NPU 与 CPU 后端接入

[English](linx-integration-review.md)

Linx 后端通过 NPU 实现和 CPU-SIM 提供原生指令 API。NPU 后端定位为**受限实验接入**：开放场景在 `gfrun` 功能模型上执行，已知不支持路径在编译时明确拒绝。CPU-SIM 在主机上执行相同的指令调用。功能模型结果不代表物理 NPU 验证。

## 依赖与来源

| 组件 | 版本 |
| --- | --- |
| Linx-TileOP-API (`linx`) | `74d7326c521aeb1b21c52602b67c55b42b3563a3` |
| SuperScalarModel / gfrun | `637bde0ec63955c4796828b297e16a057efad3cc` |
| LinxV5 Clang 15.0.4 | `b19e8c0e184cd7e3a1fa5ab57b368cbf978c3aa8` |
| PTO-spec 参考 | `42067d0f2890b8c013a868bd1247ebc6437b3179` |
| 主机 | aarch64 Linux / GCC 15.2.1 |

原仓入口的 20 个传递依赖和可选 `jcore/utils.hpp` 共构成 21 个导入头文件。原始 SHA-256 保存在 [upstream.json](../tests/cpu/linx/upstream.json)。

导入保留原生函数名、命名空间、模板、默认参数、返回值、硬件编码和汇编实现。机械变换覆盖许可注释、include 路径、保护宏、后端选择、CPU carrier 适配和带标记的准入诊断，由导入脚本重建并检查。

CPU 的 `native_api.hpp`、`region.hpp` 和 `predicate.hpp` 从原生声明生成，将内部 emitter 替换为主机实现。生成时仅移除带标记的 device 准入检查，保留上游合法性检查和退休接口的拒绝约束。源函数定义和汇编位置包含 helper 与宏，其数量不代表指令覆盖数。

## 接入架构

| 组件 | 职责 |
| --- | --- |
| `include/pto/cpu/target_config.hpp`、`include/pto/pto-inst.hpp`、公共 type/tile 头文件 | 集中选择 Linx CPU 或 device，拒绝混用目标，保留 Ascend 默认行为 |
| `include/pto/npu/linx/impl` | 提供固定版本的原生 API 和指令实现 |
| `include/pto/npu/linx/experimental.hpp` | 定义不支持路径的诊断，启用 NDEBUG 后仍然有效 |
| `include/pto/cpu/linx` | 在主机上实现原生数值格式、Tile 存储与布局、索引操作、矩阵操作、Shared 发布和 PE 参与语义 |
| `tests/script/sync_linx_headers.py` | 重现导入、插入准入检查、生成 CPU 声明，并检查来源、签名和 emitter 映射 |
| `tests/cpu/linx_instruction_cases.py`、`instruction_catalog.json` | 检查两种公共入口，将普通、显式维度和 associated 指令名关联到测试场景 |
| `tests/cpu/test_linx_models.py` | 编译并执行场景，检查诊断、独立 golden、canary 和模型完成标志 |
| `tests/cpu/test_linx_admission.py` | 在不同优化级别、NDEBUG 和公共/内部头入口下检查明确拒绝 |
| `tests/cpu/test_linx_native.py` | 执行 CPU、非法输入、目标选择、sanitizer 和上游 fixture 编译检查 |
| `tests/run_cpu.py`、`tests/script/all_cpu_tests.py` | 将 Linx 作为 standalone 套件注册到 CPU 测试流程 |

## 指令支持

| 统计组 | CPU：全部列出场景通过 | NPU 模型：全部列出场景通过 |
| --- | ---: | ---: |
| 普通指令调用名 | 151/151 | 109/151 |
| 显式维度 `_U` | 11/11 | 11/11 |
| Associated `_ASS` | 64/64 | 1/64 |
| 合计活跃指令接口名 | **226/226** | **121/226** |

229 个 CPU 场景全部通过。接入的 NPU 后端成功执行 122 个场景，明确拒绝 107 个场景。226 个活跃接口名中，105 个全部或部分受限。统计仅对应所列场景，不表示穷尽 dtype、shape、布局或所有重载。明确拒绝与数值通过分开统计，API 名称也不等于独立硬件 opcode。[指令支持矩阵](linx-tile-instruction-support_zh.md) 列出了每个名称、限制和场景。

Golden 独立于 CPU 输出构造；模型诊断、guest 完成标志、输出和 canary 共同参与判定。MX ACC 与 BIAS 独立执行；原子操作同时检查旧值与 GM 更新；reload 使用不同于初始数据的新输入。标量/Predicate 辅助接口、helper 和退休接口单独统计。

## 数值契约与开发范围

同类型 `TEXPDIF` 在指数运算前按操作类型舍入中间减法。CPU `TPACK_ASS` 读取当前 writer 的源 raw word。FP16/BF16 TEXPDIF、TRSQRT 每种格式各有 128 个差异输入；这些场景和独立 TPACK_ASS 场景在 CPU、接入的 NPU 及固定上游实现三路执行通过。

实验 CPU-SIM 的 `TRSQRT` 跟随固定模型的 typed SQRT→DIV 契约。固定规格参考采用高精度倒平方根后最终编码，可能产生不同结果。规格对齐、更多重载与边界覆盖以及物理硬件验证仍属于待开发工作。

CPU 检查单 Tile 容量、有效 shape、范围、索引对齐及 Shared 发布/组装。全芯片资源、调度/cache、页表、完整未定义值跟踪和硬件竞争顺序不在模拟范围内。原生 TPUSH/TPOP 不可用。完整约束见[后端说明](../include/pto/npu/linx/README_zh.md)。

开放受限 NPU 路径需要固定 API/编译器/模型组合，并通过对应的独立数值与内存检查，再同步调整该路径的诊断和中英文支持矩阵。

## 本地验证与复现

| 检查项 | 结果 |
| --- | --- |
| Standalone 回归，包含 NDEBUG 和 ASan/UBSan | 62/62 通过 |
| CPU 目标选择 | 5/5 通过 |
| 受限场景在 `-O0/-O2`、NDEBUG、公共/内部头入口下编译 | 428/428 明确拒绝，无编译器内部崩溃 |
| 185 个上游编译 fixture | 91 个原生语法检查通过、57 个实验范围明确拒绝、37 个上游拒绝 |
| 头文件来源检查 | 21 个导入头和 3 个生成头通过 |

编译 fixture 衡量接口接受情况，与指令数值支持分开统计。指令执行和准入检查的复现命令见[支持矩阵](linx-tile-instruction-support_zh.md#复跑)。

```bash
python3 tests/cpu/test_linx_native.py --compiler /path/to/g++-15 \
  --linx-compiler /path/to/linx-clang++ --upstream /path/to/Linx-TileOP-API \
  --survey --sanitize --release --output-dir /tmp/linx-regression
python3 tests/script/sync_linx_headers.py --upstream /path/to/Linx-TileOP-API
```

测试结果目录保存源码哈希、完整命令和执行证据。Pre-commit/OAT 检查通过。MkDocs 构建有 27 条导航警告，这些警告导致 strict 模式不能通过。工程检查与指令数值验收分别报告。
