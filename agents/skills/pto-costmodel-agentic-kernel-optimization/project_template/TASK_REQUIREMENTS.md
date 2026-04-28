# Task Requirements

这里写当前任务的附加要求。

可以写的内容例如：

- 指定默认测试 shape，例如 `BNSD=1,16,8192,512`
- 指定必须优先优化的输入规模
- 指定不允许修改的文件或模块
- 指定必须保留的行为约束
- 指定本次任务特别关注的性能指标

示例：

```text
- 把默认的 size 改成 BNSD=1,16,8192,512
- 优先关注 torch_npu baseline 对比
- 不要改动与 flash_attention_score 无关的算子
```

## Hints

<!-- Add directives below, for example:
- Operator-specific constraints or banned settings
- Remote host caveats
- Strategies to try or avoid
- Agent behavior controls
- Dependency policies (e.g., "Do not install any packages") -->

- Prefer small, isolated kernel changes. Treat correctness failure as a hard stop for that iteration.
- If 3 consecutive iterations show no improvement, re-read `../../remote.md`, inspect previous rounds in `ITERATIONS.md`, search online for additional optimization ideas, and re-evaluate the tuning direction before continuing.
