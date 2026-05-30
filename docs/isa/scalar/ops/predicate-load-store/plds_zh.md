# pto.plds

从 UB 连续加载完整谓词。

## 语法

### PTO 汇编形式

```text
plds %mask, %ub_ptr : !pto.mask<G>, !pto.ptr<i64, ub>
```

### AS Level 1（SSA）

```mlir
%mask = pto.plds %ub_ptr : !pto.ptr<i64, ub> -> !pto.mask<G>
```

### AS Level 2（DPS）

```mlir
pto.plds ins(%ub_ptr : !pto.ptr<i64, ub>) outs(%mask : !pto.mask<G>)
```

## 关键约束

- 输入与结果类型必须一致。


## 相关页面

- [谓词加载存储](../../predicate-load-store_zh.md)
