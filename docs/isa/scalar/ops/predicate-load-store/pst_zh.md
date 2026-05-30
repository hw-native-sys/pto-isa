# pto.pst

以寄存器偏移把完整谓词写回 UB。

## 语法

### PTO 汇编形式

```text
pst %mask, %ub_ptr[%areg], "DIST" : !pto.mask<G>, !pto.ptr<i64, ub>, i32
```

### AS Level 1（SSA）

```mlir
pto.pst %mask, %ub_ptr, %areg, "DIST" : !pto.mask<G>, !pto.ptr<i64, ub>, i32
```

### AS Level 2（DPS）

```mlir
pto.pst ins(%mask, %ub_ptr, %areg, "DIST" : !pto.mask<G>, !pto.ptr<i64, ub>, i32)
```

## 关键约束

- 输入与结果类型必须一致。


## 相关页面

- [谓词加载存储](../../predicate-load-store_zh.md)
