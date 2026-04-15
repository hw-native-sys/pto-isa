# pto.vldus

使用对齐状态执行非对齐向量加载。

## 语法

```mlir
%result, %align_out, %base_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 性能

### 时延与吞吐披露

`pto.vldus` 当前可公开对齐的 VPTO 时序来源是 `~/visa.txt` 与 `PTOAS` `feature_vpto_backend` 分支上的 `docs/vpto-spec.md`。
这些来源**没有**给出 `pto.vldus` 的独立数字时延，但对由 `pto.vldas` 引导的非对齐加载流给出了吞吐约束，而 `pto.vldus` 就运行在该流中。

| 指标 | 取值 | 来源依据 |
|------|------|----------|
| A5 独立时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 与匹配 `pto.vldas` 配对后的流吞吐 | 每条后续非对齐加载指令达到 one-CPI | `visa.txt` §7.5 `VLDAS` / §7.7 `VLDUS` |

当文档或调度需要引用这一吞吐结论时，应把它理解为**已初始化非对齐加载流**的性质，而不是 `pto.vldus` 单独一条指令的独立时延保证。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)
