# pto.vexp

逐 lane 取指数。

## 语法

### PTO 汇编形式

```text
%result = vexp %input, %mask : !pto.vreg<NxT>, !pto.mask
```

### AS Level 1（SSA）

```mlir
%result = pto.vexp %input, %mask : (!pto.vreg<NxT>, !pto.mask) -> !pto.vreg<NxT>
```

### AS Level 2（DPS）

```mlir
pto.vexp ins(%input, %mask : !pto.vreg<NxT>, !pto.mask)
          outs(%result : !pto.vreg<NxT>)
```

## 关键约束

- 输入与结果类型必须一致。
- 谓词宽度必须与目标向量宽度一致。
- 当前 target profile 可能对该形式施加额外限制。

## 性能

### 时延与吞吐披露

PTO 微指令页面当前使用的时序来源是 `~/visa.txt` 与最新抓取的 `PTOAS/docs/vpto-spec.md`（`feature_vpto_backend` 分支）。
对于 `pto.vexp`，这些公开来源说明了指令语义、操作数合法性和流水线位置，但**没有**发布数字时延或稳态吞吐。

| 指标 | 状态 | 来源依据 |
|------|------|----------|
| A5 时延 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |
| 稳态吞吐 | 公开来源未给出 | `visa.txt`、`PTOAS/docs/vpto-spec.md` |

如果软件调度或性能建模依赖 `pto.vexp` 的确切成本，必须在具体 backend 上实测，而不能从当前公开手册里推导出一个并未公布的常数。

## 相关页面

- [一元向量指令集](../../unary-vector-ops_zh.md)
