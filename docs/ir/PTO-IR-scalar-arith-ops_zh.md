# 标量算术操作

本文档描述来自 MLIR `arith` 方言的标量算术操作。

**重要：** PTO AS 仅支持 arith 方言的**标量操作**。不支持向量和张量操作。

**操作总数：** 47

**注意：** 操作 `arith.scaling_extf` 和 `arith.scaling_truncf` 未包含在内，因为它们在 PTO AS 中不受支持。

---

## 整数算术操作

### arith.addi - 整数加法

**描述：** 整数加法，可选溢出标志。

**语法：**
```mlir
%result = arith.addi %lhs, %rhs : i32
%result = arith.addi %lhs, %rhs overflow<nsw, nuw> : i32
```

**示例：**
```mlir
// 标量加法
%a = arith.addi %b, %c : i64
```

---

### arith.subi - 整数减法

**语法：**
```mlir
%result = arith.subi %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量减法
%a = arith.subi %b, %c : i32
```

---

### arith.muli - 整数乘法

**语法：**
```mlir
%result = arith.muli %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量乘法
%a = arith.muli %b, %c : i64
```

---

### arith.divsi - 有符号整数除法

**描述：** 有符号整数除法，向零舍入。

**语法：**
```mlir
%result = arith.divsi %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量有符号除法
%a = arith.divsi %b, %c : i32
```

---

### arith.divui - 无符号整数除法

**语法：**
```mlir
%result = arith.divui %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量无符号除法
%a = arith.divui %b, %c : i32
```

---

### arith.remsi - 有符号整数取余

**语法：**
```mlir
%result = arith.remsi %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量有符号取余
%a = arith.remsi %b, %c : i32
```

---

### arith.remui - 无符号整数取余

**语法：**
```mlir
%result = arith.remui %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量无符号取余
%a = arith.remui %b, %c : i32
```

---

### arith.ceildivsi - 向上取整除法（有符号）

**描述：** 有符号整数除法，向正无穷舍入。

**语法：**
```mlir
%result = arith.ceildivsi %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量向上取整除法
%a = arith.ceildivsi %b, %c : i32
```

---

### arith.ceildivui - 向上取整除法（无符号）

**语法：**
```mlir
%result = arith.ceildivui %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量向上取整除法（无符号）
%a = arith.ceildivui %b, %c : i32
```

---

### arith.floordivsi - 向下取整除法（有符号）

**描述：** 有符号整数除法，向负无穷舍入。

**语法：**
```mlir
%result = arith.floordivsi %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量向下取整除法
%a = arith.floordivsi %b, %c : i32
```

---

## 浮点算术操作

### arith.addf - 浮点加法

**语法：**
```mlir
%result = arith.addf %lhs, %rhs : f32
%result = arith.addf %lhs, %rhs fastmath<fast> : f32
```

**示例：**
```mlir
// 标量加法
%a = arith.addf %b, %c : f64
```

---

### arith.subf - 浮点减法

**语法：**
```mlir
%result = arith.subf %lhs, %rhs : f32
```

**示例：**
```mlir
// 标量减法
%a = arith.subf %b, %c : f32
```

---

### arith.mulf - 浮点乘法

**语法：**
```mlir
%result = arith.mulf %lhs, %rhs : f32
```

**示例：**
```mlir
// 标量乘法
%a = arith.mulf %b, %c : f64
```

---

### arith.divf - 浮点除法

**语法：**
```mlir
%result = arith.divf %lhs, %rhs : f32
```

**示例：**
```mlir
// 标量除法
%a = arith.divf %b, %c : f32
```

---

### arith.remf - 浮点取余

**语法：**
```mlir
%result = arith.remf %lhs, %rhs : f32
```

**示例：**
```mlir
// 标量取余
%a = arith.remf %b, %c : f64
```

---

### arith.negf - 浮点取负

**语法：**
```mlir
%result = arith.negf %operand : f32
```

**示例：**
```mlir
// 标量取负
%a = arith.negf %b : f32
```

---

## 位运算操作

### arith.andi - 按位与

**语法：**
```mlir
%result = arith.andi %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量按位与
%a = arith.andi %b, %c : i32
```

---

### arith.ori - 按位或

**语法：**
```mlir
%result = arith.ori %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量按位或
%a = arith.ori %b, %c : i64
```

---

### arith.xori - 按位异或

**语法：**
```mlir
%result = arith.xori %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量按位异或
%a = arith.xori %b, %c : i32
```

---

## 移位操作

### arith.shli - 左移

**语法：**
```mlir
%result = arith.shli %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量左移
%a = arith.shli %b, %c : i32
```

---

### arith.shrsi - 算术右移（有符号）

**描述：** 算术右移（符号扩展）。

**语法：**
```mlir
%result = arith.shrsi %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量算术右移
%a = arith.shrsi %b, %c : i32
```

---

### arith.shrui - 逻辑右移（无符号）

**描述：** 逻辑右移（零扩展）。

**语法：**
```mlir
%result = arith.shrui %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量逻辑右移
%a = arith.shrui %b, %c : i32
```

---

## 比较操作

### arith.cmpi - 整数比较

**描述：** 使用指定谓词比较两个整数。

**语法：**
```mlir
%result = arith.cmpi <predicate>, %lhs, %rhs : i32
```

**谓词：**
- 有符号：`slt`、`sle`、`sgt`、`sge`
- 无符号：`ult`、`ule`、`ugt`、`uge`
- 相等性：`eq`、`ne`

**示例：**
```mlir
// 标量比较
%cmp = arith.cmpi slt, %a, %b : i32
%eq = arith.cmpi eq, %x, %y : i64
```

---

### arith.cmpf - 浮点比较

**描述：** 使用指定谓词比较两个浮点数。

**语法：**
```mlir
%result = arith.cmpf <predicate>, %lhs, %rhs : f32
```

**谓词：**
- 有序：`oeq`、`one`、`olt`、`ole`、`ogt`、`oge`、`ord`
- 无序：`ueq`、`une`、`ult`、`ule`、`ugt`、`uge`、`uno`
- 总是：`true`、`false`

**示例：**
```mlir
// 标量比较
%cmp = arith.cmpf olt, %a, %b : f32
%eq = arith.cmpf oeq, %x, %y : f64
```

---

## 最小/最大操作

### arith.minsi - 最小值（有符号整数）

**语法：**
```mlir
%result = arith.minsi %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量最小值
%min = arith.minsi %a, %b : i32
```

---

### arith.minui - 最小值（无符号整数）

**语法：**
```mlir
%result = arith.minui %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量最小值（无符号）
%min = arith.minui %a, %b : i32
```

---

### arith.maxsi - 最大值（有符号整数）

**语法：**
```mlir
%result = arith.maxsi %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量最大值
%max = arith.maxsi %a, %b : i32
```

---

### arith.maxui - 最大值（无符号整数）

**语法：**
```mlir
%result = arith.maxui %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量最大值（无符号）
%max = arith.maxui %a, %b : i32
```

---

### arith.minimumf - 最小值（浮点，传播 NaN）

**语法：**
```mlir
%result = arith.minimumf %lhs, %rhs : f32
```

**示例：**
```mlir
// 标量最小值（传播 NaN）
%min = arith.minimumf %a, %b : f32
```

---

### arith.maximumf - 最大值（浮点，传播 NaN）

**语法：**
```mlir
%result = arith.maximumf %lhs, %rhs : f32
```

**示例：**
```mlir
// 标量最大值（传播 NaN）
%max = arith.maximumf %a, %b : f32
```

---

### arith.minnumf - 最小值（浮点，忽略 NaN）

**语法：**
```mlir
%result = arith.minnumf %lhs, %rhs : f32
```

**示例：**
```mlir
// 标量最小值（忽略 NaN）
%min = arith.minnumf %a, %b : f64
```

---

### arith.maxnumf - 最大值（浮点，忽略 NaN）

**语法：**
```mlir
%result = arith.maxnumf %lhs, %rhs : f32
```

**示例：**
```mlir
// 标量最大值（忽略 NaN）
%max = arith.maxnumf %a, %b : f64
```

---

## 类型转换操作

### arith.extsi - 符号扩展

**描述：** 将整数符号扩展到更宽的类型。

**语法：**
```mlir
%result = arith.extsi %in : i32 to i64
```

**示例：**
```mlir
// 标量符号扩展
%wide = arith.extsi %narrow : i32 to i64
```

---

### arith.extui - 零扩展

**描述：** 将整数零扩展到更宽的类型。

**语法：**
```mlir
%result = arith.extui %in : i32 to i64
```

**示例：**
```mlir
// 标量零扩展
%wide = arith.extui %narrow : i32 to i64
```

---

### arith.trunci - 截断整数

**描述：** 将整数截断到更窄的类型。

**语法：**
```mlir
%result = arith.trunci %in : i64 to i32
```

**示例：**
```mlir
// 标量截断
%narrow = arith.trunci %wide : i64 to i32
```

---

### arith.extf - 扩展浮点

**描述：** 将浮点数扩展到更宽的类型。

**语法：**
```mlir
%result = arith.extf %in : f32 to f64
```

**示例：**
```mlir
// 标量浮点扩展
%double = arith.extf %single : f32 to f64
```

---

### arith.truncf - 截断浮点

**描述：** 将浮点数截断到更窄的类型。

**语法：**
```mlir
%result = arith.truncf %in : f64 to f32
```

**示例：**
```mlir
// 标量浮点截断
%single = arith.truncf %double : f64 to f32
```

---

### arith.sitofp - 有符号整数转浮点

**语法：**
```mlir
%result = arith.sitofp %in : i32 to f32
```

**示例：**
```mlir
// 标量整数转浮点
%fp = arith.sitofp %int : i32 to f32
```

---

### arith.uitofp - 无符号整数转浮点

**语法：**
```mlir
%result = arith.uitofp %in : i32 to f32
```

**示例：**
```mlir
// 标量无符号整数转浮点
%fp = arith.uitofp %uint : i32 to f32
```

---

### arith.fptosi - 浮点转有符号整数

**描述：** 将浮点数转换为有符号整数（向零舍入）。

**语法：**
```mlir
%result = arith.fptosi %in : f32 to i32
```

**示例：**
```mlir
// 标量浮点转整数
%int = arith.fptosi %fp : f32 to i32
```

---

### arith.fptoui - 浮点转无符号整数

**语法：**
```mlir
%result = arith.fptoui %in : f32 to i32
```

**示例：**
```mlir
// 标量浮点转无符号整数
%uint = arith.fptoui %fp : f32 to i32
```

---

### arith.bitcast - 位转换

**描述：** 将位重新解释为不同类型（相同位宽）。

**语法：**
```mlir
%result = arith.bitcast %in : f32 to i32
```

**示例：**
```mlir
// 标量位转换
%bits = arith.bitcast %fp : f32 to i32
```

---

### arith.index_cast - 索引转换（有符号）

**描述：** 在 `index` 类型和整数类型之间转换（符号扩展）。

**语法：**
```mlir
%result = arith.index_cast %in : i32 to index
%result = arith.index_cast %in : index to i64
```

**示例：**
```mlir
// 标量索引转换
%idx = arith.index_cast %int : i32 to index
%int = arith.index_cast %idx : index to i64
```

---

### arith.index_castui - 索引转换（无符号）

**描述：** 在 `index` 类型和整数类型之间转换（零扩展）。

**语法：**
```mlir
%result = arith.index_castui %in : i32 to index
```

**示例：**
```mlir
// 标量索引转换（无符号）
%idx = arith.index_castui %uint : i32 to index
```

---

## 特殊操作

### arith.select - 条件选择

**语法：**
```mlir
%result = arith.select %condition, %true_value, %false_value : i32
```

**示例：**
```mlir
// 标量选择
%result = arith.select %cond, %a, %b : i32
%fp_result = arith.select %cond, %x, %y : f32
```

---

### arith.constant - 常量值

**语法：**
```mlir
%result = arith.constant <value> : <type>
```

**示例：**
```mlir
// 标量常量
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i64
%pi = arith.constant 3.14159 : f32
%true = arith.constant true
```

---

## 扩展算术操作

### arith.addui_extended - 扩展无符号加法

**描述：** 带溢出标志的无符号加法。

**语法：**
```mlir
%sum, %overflow = arith.addui_extended %lhs, %rhs : i32, i1
```

**示例：**
```mlir
// 标量扩展加法
%sum, %overflow = arith.addui_extended %a, %b : i32, i1
```

---

### arith.mulsi_extended - 扩展有符号乘法

**描述：** 有符号乘法，返回低位和高位。

**语法：**
```mlir
%low, %high = arith.mulsi_extended %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量扩展乘法
%low, %high = arith.mulsi_extended %a, %b : i32
```

---

### arith.mului_extended - 扩展无符号乘法

**语法：**
```mlir
%low, %high = arith.mului_extended %lhs, %rhs : i32
```

**示例：**
```mlir
// 标量扩展乘法（无符号）
%low, %high = arith.mului_extended %a, %b : i32
```
