# pto.vldas

初始化非对齐加载所需的对齐状态。

## 语法

```mlir
%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 性能

### 时延与吞吐披露

`pto.vldas` 当前可公开对齐的 VPTO 时序来源是 `~/visa.txt` 与 `PTOAS` `feature_vpto_backend` 分支上的 `docs/vpto-spec.md`。
这些来源**没有**给出 `pto.vldas` 本身的独立数字时延，但 `visa.txt` 对它所引导的非对齐加载流给出了明确吞吐约束。

| 指标 | 取值 | 来源依据 |
|------|------|----------|
| A5 初始化指令时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 后续非对齐加载吞吐 | 同一流中的后续非对齐加载指令达到 one-CPI | `visa.txt` §7.5 `VLDAS` |

因此，`pto.vldas` 的公开时序语义应理解为**流初始化指令**：已披露的是后续非对齐加载流的吞吐，而不是该 setup 指令自身的独立周期常数。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)
