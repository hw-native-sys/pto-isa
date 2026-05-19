# pto.pld

以寄存器偏移从 UB 加载完整谓词。

## 语法

### PTO 汇编形式

```text
pld %mask, %ub_ptr[%areg], "DIST" : !pto.mask<G>, !pto.ptr<i64, ub>, i32
```

### AS Level 1（SSA）

```mlir
%mask = pto.pld %ub_ptr, %areg, "DIST" : !pto.ptr<i64, ub>, i32 -> !pto.mask<G>
```

### AS Level 2（DPS）

```mlir
pto.pld ins(%ub_ptr, %areg, "DIST" : !pto.ptr<i64, ub>, i32) outs(%mask : !pto.mask<G>)
```

## 关键约束

- 输入与结果类型必须一致。


## 相关页面

- [谓词加载存储](../../predicate-load-store_zh.md)
