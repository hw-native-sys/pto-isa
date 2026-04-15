# pto.vstus

执行流式非对齐向量存储。

## 语法

```mlir
%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

## 关键约束

- UB 地址空间与对齐要求必须满足。
- DMA 与向量计算之间的顺序边必须显式建立。

## 性能

### 时延与吞吐披露

`pto.vstus` 当前可公开对齐的 VPTO 时序来源是 `~/visa.txt` 与 `PTOAS` `feature_vpto_backend` 分支上的 `docs/vpto-spec.md`。
这些来源详细定义了有状态非对齐 store 的缓冲语义，但**没有**公布 `pto.vstus` 的数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt` §8.9 `VSTUS`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt` §8.9 `VSTUS`、`PTOAS/docs/vpto-spec.md` |

由于 `pto.vstus` 参与的是带状态的缓冲写流，只要公开 ISA 来源还没有给出数字时序，就必须把它视为 backend-specific timing。

## 相关页面

- [向量加载存储指令集](../../vector-load-store_zh.md)
