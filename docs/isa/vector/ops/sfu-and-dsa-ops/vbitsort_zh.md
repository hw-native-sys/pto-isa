# pto.vbitsort

执行位级排序。

## 语法

### PTO 汇编形式

```asm
vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, index
```

### AS Level 1（SSA）

```mlir
pto.vbitsort %dest, %src, %indices, %repeat_times
    : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<i32, ub>, index
```

## 关键约束

- 某些操作只在特定 profile 或元素类型上受支持。
- lowering 不能把专用语义退化为未文档化的普通算术组合。

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vbitsort`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vbitsort` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- [SFU 与 DSA 指令集](../../sfu-and-dsa-ops_zh.md)
