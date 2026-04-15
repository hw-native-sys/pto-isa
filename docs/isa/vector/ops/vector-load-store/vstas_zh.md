# pto.vstas

初始化流式非对齐存储状态。

## 语法

```mlir
pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 性能

### 时延与吞吐披露

`pto.vstas` 当前可公开对齐的 VPTO 时序来源是 `~/visa.txt` 与 `PTOAS` `feature_vpto_backend` 分支上的 `docs/vpto-spec.md`。
这些来源把带尾部缓冲的 flush 语义写得很清楚，但**没有**公布 `pto.vstas` 的数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt` §8.6 `VSTAS`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt` §8.6 `VSTAS`、`PTOAS/docs/vpto-spec.md` |

如果代码调度依赖尾刷写步骤的成本，必须在具体 backend 上实测，不能把公开 ISA 文本当作已经给出固定周期常数。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)
