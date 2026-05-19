# pto.pstu

以流式形式把谓词写回 UB。

## 语法

### PTO 汇编形式

```text
pstu %align_in, %mask, %base_in : !pto.align, !pto.mask<G>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

### AS Level 1（SSA）

```mlir
%align_out, %base_out = pto.pstu %align_in, %mask, %base_in : !pto.align, !pto.mask<G>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>
```

### AS Level 2（DPS）

```mlir
pto.pstu ins(%align_in, %mask, %base_in : !pto.align, !pto.mask<G>, !pto.ptr<T, ub>)
       outs(%align_out, %base_out : !pto.align, !pto.ptr<T, ub>)
```

## 关键约束

- UB 地址空间和对齐要求必须满足。
- 谓词搬运覆盖完整谓词宽度。

## 相关页面

- [谓词加载存储](../../predicate-load-store_zh.md)
