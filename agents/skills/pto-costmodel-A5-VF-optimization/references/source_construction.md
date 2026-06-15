# Baseline 与源码构造

当任务没有提供可工作的 CCE/PTO/DSL baseline kernel，或优化需要使用不熟悉的 PTO ISA / CCE intrinsic 语法时，读取本文件。

## 先 Baseline，后优化

如果没有 baseline，先构造简单、correctness-first 的 baseline，再进行任何性能调优。

baseline 要求：

- 实现任务要求的 workload 和 kernel signature
- 使用目标仓库和工具链版本支持的 PTO-ISA、CCE 或 DSL 语法
- 有 golden correctness check
- 有可复现的 build/run command
- 记录初始任务声明的性能 metric；如果 trace 暴露 VF timing，也记录 VF timing
- 作为 round 0 或第一个 baseline artifact 保存

当 baseline 语义仍不确定时，不要开始优化。慢但正确的 baseline 优于快但未经验证的实现。

## PTO/CCE 查询顺序

当不确定 PTO 指令语法、CCE intrinsic 语法、语义、mask、mode 或目标可用性时，按以下顺序查询：

1. 仓库内 PTO-ISA 文档和头文件，尤其是 `docs/isa/`、`docs/coding/`、`docs/machine/` 和 `include/pto/`。
2. 仓库中覆盖相同指令或数据流的 tests 和 examples。
3. 仓库中的 costmodel 文档或头文件，如果存在，例如 `docs/costmodel*` 和 `include/pto/costmodel/`。
4. 当编写直接 CCE intrinsic 时，查询 Ascend CCE intrinsic 官方 API 文档：
   `https://www.hiascend.com/document/detail/zh/canncommercial/850/API/cceintrinsicapi/cceapi_0024.html`
5. 当前 CANN/toolchain 版本对应的本地 CANN compiler headers，例如：
   `ascend-toolkit/cann-9.0.0-beta.1/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h`
6. skill 自带 CCE ISA 参考：`references/isa.md`，主要用于直接 CCE vector intrinsic 场景。
7. 如果当前任务提供额外 ISA 证据，可作为补充上下文。
8. 如果指令仍不明确，在依赖它之前询问用户。

当仓库文档、本地 header 和网页文档不一致时，优先以当前仓库/工具链实际可 build 行为为准，并在 round log 中记录差异。

## Baseline 设计原则

- 优先使用直接的数据流和显式 UB layout。
- 除非 lifetime reuse 已证明，否则保持 UB 区域不重叠。
- 先保守添加必要同步；只有 correctness 建立后再优化 barrier。
- 在可行时使用 hardware-loop-friendly loop form：简单 unsigned loop variable、静态 bound、VF scope 内无不必要 branch。
- baseline 通过 correctness 前，避免使用巧妙但难验证的指令替换。

## Addressing Mode 与 POST_UPDATE

部分 vector load/store intrinsic 支持 post-update addressing form。使用前，先确认当前 CANN header 或官方文档中的精确签名。记录 pointer operand 是否更新、stride 单位是什么，以及 mask/mode 是否影响 overload resolution。

除非简单正确实现必须使用，否则不要在 baseline 中引入 post-update addressing。优先在 conventional-addressing baseline 通过 correctness 后，把它作为单独优化轮次。

## Baseline 日志项

baseline entry 应包含：

- source path 和 kernel name
- workload shape 与 inputs/outputs
- build flags、runner 和 target
- correctness result
- 任务声明的 performance metric
- first VF start、last VF end、total VF cycles，如果可用
- per-VF execute time 和 instruction count，如果可用
- 已知限制或保守选择
