# AKO4PTO Task — PTO 算子优化

适用于 **PTO-DSL**（Python）和 **PTO-tile-lib**（C++）两种场景的 PTO 算子优化任务。

## Skill 关系

本 skill 依赖同级目录下的 **PTO Costmodel 查询 skill**：

- **`pto-costmodel-query-pto-cycles`** — 查询单条 PTO ISA 指令的仿真 cycles 数
  - 路径：`agents/skills/pto-costmodel-query-pto-cycles/SKILL.md`
  - 用途：在优化迭代中，通过 costmodel 分析不同 tile shape / 参数配置下 PTO 指令的 cycle 开销，指导优化方向
  - 典型用法：`python3 tools/query_pto_cycles.py TMATMUL half half float --m 128 --k 64 --n 128`
  - 能力：单条 PTO ISA 指令级 cycles 仿真（TMATMUL、TADD、TEXP 等），纯 CPU 运行，不需真实硬件
  - 局限：不支持算子级（多指令组合）仿真

本 skill（`pto-costmodel-agentic-kernel-optimization`）在迭代过程中会调用 costmodel 来：
1. 对比不同 tile 参数下的指令 cycle 开销
2. 验证优化假设（如"更大 tile 的 TMATMUL 每 FLOP cycle 更低"）
3. 指导参数搜索方向，减少盲目试错

## 适用范围

| 维度 | PTO-DSL (Python) | PTO-tile-lib (C++) |
|------|------------------|---------------------|
| 构建方式 | Python JIT (`wrapper._build()`) | CMake + `make -j16` + Bisheng |
| 调参方式 | Python 环境变量 / kernel 参数 | C++ 模板参数、`.h` 常量、`.cpp` 内部参数 |
| Benchmark | Python benchmark 脚本 → `report.json` | 编译二进制 → `report.csv` (duration_us, TFLOPS) |
| 代码位置 | `workspace/pto-kernels/` 下 `.py` | `workspace/kernel/` 下 `.cpp`/`.hpp` |
| 正确性 | `report.json` 中 `correctness.passes` | 内置 `ResultCmp` 输出 "test success"/"test failed" |
| 典型算子 | `flash_attention_score`, `ffn`, etc. | `fa_performance`, etc. |

Agent 会根据用户提供的算子源码自动判断属于哪种场景，并选择对应的构建和 benchmark 流程。

---

## 首要原则

PTO 算子优化**不是**通用代码整理，而是面向硬件的瓶颈消除过程。

每一次优化必须能回答以下四个问题，否则不要动手改代码：

1. 当前瓶颈是什么
2. 为什么这个改动能在 Ascend 上有效
3. 预期改善哪个指标
4. 如何保证正确性

一个好的 PTO 优化是：**正确的、可衡量的、可重复的、瓶颈驱动的、有硬件依据的、足够可维护以保留下来**。不追求巧妙、复杂或理论上优雅。

---

## 主流程

按下面顺序执行。

### 1. 确认优化算子

**必须**向用户确认要优化的算子。有两种方式：

- **用户直接给出算子源码路径**（如 `/path/to/flash_atten/`）
- **Agent 扫描已知路径**：搜索项目中或用户指定的目录，发现可优化的算子并列表让用户选择

确认算子后，Agent 应快速浏览源码，**自动判断**是 PTO-DSL 还是 PTO-tile-lib 类型：
- 含 `.py` kernel 定义 + `pto_kernels` / `ptodsl` 相关导入 → PTO-DSL
- 含 `CMakeLists.txt` + `.cpp`/`.hpp` + PTO tile 宏 → PTO-tile-lib

向用户确认判断结果。

### 2. 确认优化 Case

**必须**向用户确认本次优化的具体 case 配置：

- **Shape**: 如 `S=32768, HEAD_SIZE=128, BATCH=1`（序列长度、头维度等）
- **Dtype**: 如 `float16`、`bfloat16`
- 是否 causal mask
- 其他特殊配置

如果算子有多个可测 case，列出可选项让用户选择。

### 3. 确认远程环境和 Costmodel

向用户确认：
1. 远程服务器连接方式（使用哪个 connect skill）
2. 是否使用 costmodel，costmodel 工具路径
3. 算子在远程的部署路径（如果已有）

### 4. 自动创建项目工作区

获取以上信息后，Agent **自动**创建项目工作区，无需用户手动操作：

```bash
# 从 project_template/ 复制基础结构
mkdir -p projects/<operator_name>/
cp -r project_template/* projects/<operator_name>/

# 创建必要目录
mkdir -p projects/<operator_name>/{context,runs,workspace}

# 复制算子源码到隔离工作区
# PTO-DSL: 复制到 workspace/pto-kernels/
# PTO-tile-lib: 复制到 workspace/kernel/
cp -r <source_path> projects/<operator_name>/workspace/<pto-kernels|kernel>/
```

创建完成后：
- 根据用户提供的 case 信息，**自动生成** `TASK_REQUIREMENTS.md`（含算子名、case 配置、可调参数等）
- 向用户确认是否需要修改 `context/` 参考资料和 `TASK_REQUIREMENTS.md`

### 5. 读取任务上下文

开始执行前先阅读：
- `projects/<operator_name>/TASK_REQUIREMENTS.md`
- `projects/<operator_name>/context/`
- 远程连接 skill（`connect-server`、`remote.md` 等）
- `projects/<operator_name>/workspace/` 下所有与目标算子相关的源码

在进入调优前，至少要明确：

| 维度 | PTO-DSL | PTO-tile-lib |
|------|---------|--------------|
| 入口函数 | kernel spec 路径 | 入口模板函数（如 `LaunchTFA<...>`） |
| 可调参数 | Python 环境变量 / kernel 参数 | C++ 模板参数 / `.h` 常量 / `.cpp` 内部参数 |
| benchmark 命令 | Python benchmark 脚本 | `./fa_performance --npu=0 --cases="..."` |
| 正确性判定 | `report.json` 中 passes 字段 | "test success" / "test failed" |
| 性能指标 | `report.json` 中 duration_ms / TFLOPS | `report.csv` 中 duration_us / TFLOPS |
| 构建命令 | `wrapper._build()` | `cmake + make -j16` |

### 6. 打通远程链路

如果远程环境是首次接入，第一优先级不是调优，而是把环境带到"可以稳定跑 benchmark"。

只有满足下面条件，才进入真正的性能迭代：

**PTO-DSL**:
1. SSH 连接成功
2. 远程工作目录已建立，隔离工作区已上传
3. 已确认远程 Python 路径和版本
4. 至少成功跑出一份有效 `report.json`，确认 baseline 和 pto 都 benchmark ok + correctness passes

**PTO-tile-lib**:
1. SSH 连接成功
2. 远程工作目录已建立
3. 远程已安装 Bisheng 编译器、CANN、ACL runtime
4. 至少成功完成一次完整构建（`cmake + make`）
5. 至少成功跑出一份有效结果：构建无错误 + "test success" + `report.csv` 有有效数据

如果以上任一项不满足，本轮应视为"环境打通轮"，优先修环境，不要急着改 kernel。

### 7. 跑 torch_npu Baseline

**在开始优化迭代前**，必须先跑一份 `torch.nn.functional.scaled_dot_product_attention`（或算子对应的 PyTorch 标准实现）的 baseline 性能数据，作为最终对比的参照。

在远程服务器上运行 PyTorch benchmark：
- 使用与优化 case 相同的 shape、dtype 配置
- 至少 warmup 5 次 + 10 次计时
- 记录 avg、min、max duration 和 TFLOPS
- 记入 `ITERATIONS.md` 的 torch_npu baseline 部分

### 8. 进入性能迭代

每次新会话开始调优前，必须先：
1. 了解当前最佳结果
2. 了解最近失败的假设
3. 从当前已知最佳状态继续，不要盲目从头开始
4. 参考 `projects/<operator_name>/context/` 和 `TASK_REQUIREMENTS.md`

然后按下面的**优化规则**和**迭代协议**执行。

---

## 构建与运行流程

### PTO-DSL (Python)

```bash
# 构建
python3 -c "from pto_kernels import <kernel>; wrapper._build()"

# Benchmark
python3 bench_<kernel>.py --case "<shape_config>"
```

### PTO-tile-lib (C++)

```bash
# 构建
cd workspace/kernel/
rm -rf build && mkdir build && cd build
python3 ../scripts/generate_cases.py --cases "<case_config>" --qk-preload <N>
cmake -DRUN_MODE=npu -DSOC_VERSION=Ascend910B1 ..
make -j16

# Benchmark
python3 ../scripts/gen_data.py --cases "<case_config>"
./fa_performance --npu=0 --cases "<case_config>"
```

---

## 优化规则

### 一轮一假设

每轮迭代只测试 **一个** 假设。所有代码/参数修改必须服务于同一个假设。

### 正确性先于性能

每一轮必须按顺序执行：修改代码/参数 → 构建 → 跑正确性测试 → 只有通过后才跑性能测试。

- 正确性不通过 → 性能结果无效，记录并终止本轮
- 不允许放宽容差（除非用户明确批准）
- 不允许修改 benchmark workload 或验证逻辑来"提分"

### 可编辑范围

默认只允许修改算子 kernel 源码和紧密相关的参数配置。

除非明确有 bug，否则不得修改 benchmark 脚本、运行 harness、测试逻辑、正确性阈值、workload shapes。永远不要通过让 benchmark 变简单来"提分"。

### 强制瓶颈分类

每次改代码之前，必须将当前瓶颈归类为以下之一：

| 类别 | 说明 |
|---|---|
| GMEM traffic bound | 全局内存带宽瓶颈 |
| UB / tile buffer pressure | UB 缓冲压力 |
| small-transfer inefficiency | 小包传输低效 |
| compute under-utilization | 算力利用不足 |
| AIC/AIV imbalance | Cube/vector 不平衡 |
| pipeline bubble / poor overlap | 流水线气泡 / overlap 不足 |
| barrier / wait / sync overhead | 同步开销过大 |
| register pressure / occupancy loss | 寄存器压力 / 占用率下降 |
| bad tile shape | tile 形状不佳 |
| dependency chain too long | 依赖链过长 |
| CV FIFO pressure | Cube-Vector FIFO 拥塞 |
| unknown | 未知——必须先检查代码和 profiling 数据 |

### Ascend 硬件检查清单

每轮开始前必须逐项检查：

**Tile shape** — 分块合理性、计算单元填充率、tail 开销、UB 用量

**GM ↔ UB 搬运** — 总搬运次数、冗余 reload/store-back、连续性、复用率

**小包风险** — 碎片化 copy、每 tile 输出是否太小、shape 是否破坏传输效率

**Barrier / wait / sync** — 数量、是否不必要地串行化

**UB 压力** — tile buffer + FIFO buffer + 临时 buffer 总占用

**Pipeline overlap** — 各阶段的 overlap 是否充分

**AIC / AIV 平衡** — cube 和 vector 侧的负载是否平衡

**瓶颈转移风险** — 是否只是转移了瓶颈而非消除

### 优化优先级（从高到低）

1. **消除不必要的数据搬运** — 减少冗余 GM load/store，提高 UB 复用
2. **修正不合理的 tile shape** — 改善 tile 维度以提升利用率和搬运效率
3. **减少小包传输低效** — 合并或批量化小包传输
4. **改善 load/compute/store overlap** — 改善稳态 pipeline overlap
5. **减少 barrier/wait 开销** — 删除可证明无用的 barrier
6. **降低 UB / 寄存器压力** — 缩短 live range，减少不利于 overlap 的临时 buffer
7. **改善指令组织** — 重组计算以改善 AIC/AIV 平衡
8. **微调只在以上问题都解决后进行**

### 高风险变更

以下变更属于高风险，必须额外说明理由：

- 更大的 tile 可能导致 UB 爆炸或编译失败
- 修改 FIFO 大小可能改变 kernel 语义
- 删除 barrier 前缺少依赖证明
- 指令重排可能改变数值行为
- 没有硬件理由的"可能更快"的改动

### 禁止行为

1. 一轮中做多个不相关的改动
2. 跳过正确性测试
3. 从单次噪声测量中声称胜利
4. 修改 benchmark workload 或弱化验证逻辑来提高数字
5. 没有瓶颈证据就做大规模重写
6. 多次失败后继续随机 trial-and-error
7. 隐藏失败的实验
8. 为代码美观而优化，而非针对硬件瓶颈
9. 仅基于理论推理就宣布成功

### 三轮失败规则

如果连续 3 轮没有实质性收益，必须停止盲目的局部微调：

1. 重新检查当前瓶颈分类
2. 检查之前的轮次是否在攻击症状而非根因
3. 寻找不同的优化方向
4. 优先考虑结构性变化
5. 可以通过上网搜寻相关资料参考

---

## 迭代协议

每次"为了验证性能而进行的代码修改或参数调整"加上"一次构建和 benchmark 执行"，共同算作一轮迭代。迭代编号按顺序递增：`0, 1, 2, ...`（iter-0 为 baseline）。

每轮完成后，必须做以下事项：

1. 在 `projects/<operator_name>/runs/iter-XXX/` 下保存：
   - `files.txt` — 本轮修改的文件列表
   - `patch.diff` — 本轮代码 diff
   - `notes.md` — 本轮详细记录
2. 在 `projects/<operator_name>/ITERATIONS.md` 追加一行简约摘要

### patch.diff 记录规则

- **代码修改类迭代**：标准 `diff -u` 格式
- **调参类迭代**：
```
# 本轮为调参类迭代
# 参数变更：
-  PARAM=old_value
+  PARAM=new_value
# 基于 iter-XXX 的配置
```

### Benchmark 纪律

- **相同 workload**：前后对比必须使用相同 case
- **可重复性**：重要性能结果必须多次运行
- **噪声处理**：数据有噪声标记为 `inconclusive`
- **胜利判定**：正确性通过 + 相同 workload + 改善可重复 + 收益大于噪声

### 结果判定

- `kept`：结果正确，且优于当前已知最好结果
- `neutral`：结果正确，但没有优于当前前沿结果
- `failure`：结果错误、编译失败，或本轮实验无效
- `inconclusive`：数据波动较大，重跑确认

---

## notes.md 记录要求

### Pre-Edit（改代码前填写）

- **Kernel**：目标算子名和当前配置
- **Current bottleneck**：
- **Evidence**：
- **Hypothesis**：
- **Why it should help on Ascend**：
- **Expected improved metric**：
- **Main risk**：

### Changes（改完代码后填写）

- **修改类型**：代码修改 / 模板参数调整 / 环境变量 / 构建配置
- **修改的文件**：列出所有修改的文件路径
- **具体变更**：写明每个参数的旧值→新值，或代码修改的具体内容
- **完整参数快照**：本轮所有参数配置（方便复现）

### Post-Run（跑完 benchmark 后填写）

- **Correctness**：pass / fail
- **Performance**：写明具体数值（duration、TFLOPS 等）
- **Stability**：stable / noisy
- **Result**：kept / neutral / failure

### Analysis

详细分析为什么有效或失败。

### Next

下一轮准备尝试什么。

### 环境打通轮额外记录

远程环境版本信息、构建是否成功、首次有效结果路径、主要坑点。

---

## 收尾要求

优化结束后，**必须**生成一张总结对照图（PNG），风格参考 `projects/flash_attention/summary.png`。

### 图表结构：上下双图

**上半部分 — 提升倍数折线图**：
1. Y 轴：相对于初始 PTO 的 speedup 倍数（x）
2. 折线：每轮实际 speedup（绿色）+ best frontier（橙色）
3. 散点标记：绿色=kept、蓝色=neutral、红色叉=failure
4. **阶段背景色**：不同优化阶段用不同颜色背景区分（如 baseline / costmodel-guided / 参数验证 / fine-tune）
5. **里程碑标注**：关键提升点标注具体改动内容
6. 顶部摘要：总迭代数、kept/neutral/failure 统计、最终 speedup、与 torch_npu 的对比

**下半部分 — 绝对延迟柱状图**：
1. Y 轴：duration（us 或 ms，必要时可用 log scale）
2. 柱状图颜色区分：灰色=baseline、绿色=kept、蓝色=neutral、红色=failure
3. torch_npu 参考线（紫色虚线）
4. 每根柱标注具体数值

**底部**：文字说明 torch_npu / PTO initial / PTO best 数值和关键结论

### 迭代精简

如果总迭代数较多（>10），X 轴只展示**关键迭代**：baseline、每次 kept、典型 failure、最终 best。将连续的 neutral/failure 压缩为代表性的一两个，无需逐个展示。

保存到 `projects/<operator_name>/summary_chart.png`。
