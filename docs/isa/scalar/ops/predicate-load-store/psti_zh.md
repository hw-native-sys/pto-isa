# pto.psti

以立即数偏移把完整谓词写回 UB。

## 语法

### PTO 汇编形式

```text
psti %mask, %ub_ptr[%imm], "DIST" : !pto.mask<G>, !pto.ptr<i64, ub>, i32
```

### AS Level 1（SSA）

```mlir
pto.psti %mask, %ub_ptr, %imm, "DIST" : !pto.mask<G>, !pto.ptr<i64, ub>, i32
```

### AS Level 2（DPS）

```mlir
pto.psti ins(%mask, %ub_ptr, %imm, "DIST" : !pto.mask<G>, !pto.ptr<i64, ub>, i32)
```

## 关键约束

- 输入与结果类型必须一致。


## 相关页面

- [谓词加载存储](../../predicate-load-store_zh.md)
