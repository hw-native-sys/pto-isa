# pto.psts

把完整谓词连续写回 UB。

## 语法

### PTO 汇编形式

```text
psts %mask, %ub_ptr : !pto.mask<G>, !pto.ptr<i64, ub>
```

### AS Level 1（SSA）

```mlir
pto.psts %mask, %ub_ptr : !pto.mask<G>, !pto.ptr<i64, ub>
```

### AS Level 2（DPS）

```mlir
pto.psts ins(%mask, %ub_ptr : !pto.mask<G>, !pto.ptr<i64, ub>)
```

## 关键约束

- 输入与结果类型必须一致。


## 相关页面

- [谓词加载存储](../../predicate-load-store_zh.md)
