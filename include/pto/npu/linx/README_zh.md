# Linx 受限实验 NPU 后端与 CPU-SIM

[English](README.md)

原生 NPU 后端当前为**受限实验接入**。已知不支持路径在编译时以 `PTO Linx experimental:` 明确提示原因。选择场景前请查阅[逐指令矩阵](../../../../docs/linx-tile-instruction-support_zh.md)。本轮原生实测使用本机 gfrun 功能模型，不是物理 NPU 测试。

## 来源与维护

`impl/` 导入 Linx-TileOP-API 的 `linx` 分支固定提交 `74d7326c521aeb1b21c52602b67c55b42b3563a3`。21 个原文件的哈希在 [upstream.json](../../../../tests/cpu/linx/upstream.json)；保留上游组织、格式和实现，并增加仓库 CANN 许可注释。

允许的接入变换为许可注释、include 路径、保护宏、目标选择、既有 CPU carrier 适配及机械生成的限制检查。函数声明、参数顺序、默认值、命名空间、硬件编码和原有合法性检查保持原样。不会将 Ascend 同名调用转换成 Linx 语义。

CPU `native_api.hpp/region.hpp/predicate.hpp` 从原生声明生成，仅去除带边界标记的 device 限制，再替换执行 emitter。GCC 对无关 Local assembly 重载的提前实例化由 CPU 专用延迟类型查询处理；真实 carrier 类型不变。生成器遇到未映射汇编或签名漂移会失败。

```bash
python3 tests/script/sync_linx_headers.py
python3 tests/script/sync_linx_headers.py --upstream /path/to/Linx-TileOP-API \
  --inventory /tmp/linx-inventory.json
# 维护者审阅来源/限制变更后重新生成：
python3 tests/script/sync_linx_headers.py --upstream /path/to/Linx-TileOP-API --write
```

不要直接编辑生成声明。恢复支持需固定包含修复的 API、编译器、模型版本，并重跑独立数值场景；上游 issue 关闭本身不构成验收。没有供产品构建绕过限制的宏。

## 构建与启动

公共入口为 `<pto/pto-inst.hpp>`，C++20 或更新标准。原生编译器的 `__linx` 自动选择 Linx；显式 `PTO_NPU_ARCH_LINX` 也必须使用原生 Linx 编译器，不能与 Ascend 或 cost-model 标志混用。

```bash
linx-clang++ -std=c++20 -mlxbc -O2 -I include -c kernel.cpp -o kernel.o
g++ -std=c++20 -O2 -ffp-contract=off -pthread -I include \
  -D__CPU_SIM -DPTO_CPU_SIM_TARGET_LINX kernel.cpp host.cpp -o kernel-test
```

CPU-SIM 要求 little-endian LP64 主机，关闭 fast-math 和浮点融合。Linx-only 构建不依赖 CANN 或额外库。既有打包递归安装整个 include 目录，无需第二套构建或安装系统。

单 PE 主机调用的 `get_thread_idx()` 为 0。Shared、多 PE 矩阵或 GMOV 使用：

```cpp
pto::linx_cpu::runCore([&] { kernel(input, output); });
```

启动器默认创建四个 PE 上下文；`CoreConfig{peCount, memory}` 可选择 1–4 个 PE 和显式 GM 映射。`syncCore()` 是可复用的参与者屏障，worker 异常和分歧屏障会唤醒等待线程，在所有 worker join 后向调用方传播异常。Shared 生产者按 mask 等待参与 PE；单生产者不等待其他 PE，消费者按 generation 等待发布。GMOV 需要四个源参与者；单 PE 启动也支持自拷贝。参与线程必须按相同顺序调用对应操作，缺少选定参与者会触发 CPU 错误。绝对 Shared slot 对未执行生产指令的消费者也可见。普通 Shared load 使用单生产者，多生产者须遵守显式组装协议。

显式 slot 为每个 PE 保存消费位置，每次 acquire 取得下一次发布，避免快生产者覆盖慢消费者的数据。重复使用已取得的 generation 应复制 Shared handle。所有活跃 PE 消费后或启动结束时回收发布记录。这覆盖已测生产/消费协议，不表示实现了 TPUSH/TPOP 或有界硬件 FIFO 背压。

设备地址需要显式提供 backing storage：

```cpp
std::vector<std::byte> storage(4096);
std::array<pto::linx_cpu::MemoryRegion, 1> regions{{{0x10000, storage}}};
pto::linx_cpu::runCore([&] { kernel(); }, {4, regions});
```

Tile GM 访问转换映射地址并检查范围；未映射的普通主机指针保持主机语义，低地址必须有映射。映射及存储需保持有效直至 `runCore` 返回，重叠及溢出的地址区间会被拒绝。普通 C++ 标量解引用不会自动转换，host adapter 应显式调用 `globalAddress(pointer, offset, length)`。

## 仿真契约

CPU 检查单 Tile 物理容量、有效 shape、子视图/组装范围、字节索引范围和对齐；模拟 M16/M32/N8、M32 B64 分平面、小浮点及 packed 布局、Predicate CELL、Local/Shared 发布、peer move、Shared slot、矩阵/MX/后处理。允许别名的输入先快照再发布输出，原子主机更新串行化并返回旧值。

格式、舍入、归约和协议参考固定 PTO-spec `42067d0f2890b8c013a868bd1247ebc6437b3179`。TGATHER/TSCATTER 使用行索引，GM indexed 使用带符号字节偏移；MGATHER_MASK/MSCATTER_MASK 使用 U8 0/1 lane，不能使用 RowMajor 比较产生的 packed-bit carrier。非活动 scatter 保持原内存。

超越函数使用主机数学库。TLOG 分阶段舍入 log2 和 ln(2) 乘法；同类型 TEXPDIF 先舍入减法，扩展形式在 FP32 中减法，再进行 EXP。TRSQRT 明确遵循固定模型的 typed SQRT→DIV，包含 FP16/BF16；固定规格的高精度倒平方根后最终编码与此不同，该争议仍未闭环，不能宣称完整规格符合性。

允许的 CPU 放宽包括全芯片寄存器/Shared 总资源、bank/cache/时序、页表/内存域/故障恢复、完整未定义值和所有非法别名跟踪，以及设备原子竞争的具体顺序。Local 初始零值不承诺设备 Null padding 为零；TPREFETCH 不模拟缓存或访问权限。原生没有 TPUSH/TPOP，本接入不增加 Ascend FIFO。浮点 sticky 状态不在该 API 仿真范围。

## 验证

```bash
CXX=g++ python3 tests/run_cpu.py --testcase linx_native --verbose
python3 tests/cpu/test_linx_native.py --compiler g++ \
  --sanitize --release --output-dir /tmp/linx-cpu
python3 tests/cpu/test_linx_native.py --compiler g++ \
  --linx-compiler /path/to/linx-clang++ --upstream /path/to/Linx-TileOP-API \
  --survey --output-dir /tmp/linx-interface
python3 tests/cpu/test_linx_models.py --compiler g++ \
  --linx-compiler /path/to/linx-clang++ --model-root /path/to/SuperScalarModel \
  --runtime-dir /path/to/target-libs --upstream /path/to/Linx-TileOP-API \
  --output-dir /tmp/linx-instructions --jobs 8
```

模型和工具链准备参考 [SuperNPUBench](https://github.com/PTO-ISA/SuperNPUBench#running-on-the-models)。使用匹配的 LinxV5 Clang 15、启动 ABI 和 target musl；runner 显式处理 bundled SDK 的 include 优先级，防止误测旧头文件。

默认按指令审计：151 个普通调用名、11 个 `_U`、64 个 `_ASS`，两种入口的预处理名录必须一致。CPU 151/151，连同包装为 226/226；原生 109/151，连同包装为 121/226。一个名字需通过所有所属场景才计通过；4 个 helper、32 个退休声明和小写辅助族另列，不能称为 226 个硬件 opcode。

229 个场景 CPU 全过，两条未加限制原生路径均为 122 通过、107 未通过。受限原生保留 122 个执行通过，107 个明确拒绝。`--case NAME` 选择单例，`--legacy-cases` 回放历史 130 个场景。接口 fixture 编译结果、明确拒绝和数值通过分开记录。

完整数值审计遇到受限/失败场景仍返回非零。另行检查接入边界：

```bash
python3 tests/cpu/test_linx_admission.py \
  --baseline /tmp/unguarded-instructions/result.json \
  --restricted /tmp/restricted-instructions/result.json \
  --output-dir /tmp/linx-admission --jobs 8
```

该检查核验开放场景通过、已知失败明确拒绝，并复查 NDEBUG、-O0/-O2、公有和直接内部 include。部分诊断在模板实例化时触发，避免编译器先崩溃；其他 error 属性需代码生成才能检查，`-fsyntax-only` 不足够。诊断历史行为只使用仓库外未加限制快照（`--device-include`）或原仓。

`result.json` 和 `instruction-matrix.json` 保存源码/golden 哈希、命令、返回码、模型诊断、所属场景和结果。通过需同时满足完成状态、独立 golden、末尾 canary 和逐字节一致；rc=0 不是通过。详细边界见[指令矩阵](../../../../docs/linx-tile-instruction-support_zh.md)，修改及回归见[审阅记录](../../../../docs/linx-integration-review_zh.md)。
