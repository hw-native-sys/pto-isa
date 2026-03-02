# 控制流操作

本文档描述来自 MLIR `scf`（结构化控制流）方言的结构化控制流操作。

**操作总数：** 7

---

## 循环操作

### scf.for - For 循环

**描述：** 带有下界、上界和步长的 for 循环。支持循环携带变量和有符号/无符号比较。

**语法：**
```mlir
scf.for %iv = %lb to %ub step %step {
  // 循环体
}

scf.for %iv = %lb to %ub step %step 
    iter_args(%arg = %init) -> (type) {
  // 带循环携带变量的循环体
  scf.yield %new_value : type
}

scf.for unsigned %iv = %lb to %ub step %step : i32 {
  // 无符号比较
}
```

**操作数：**
- `lb`: 下界（索引或整数）
- `ub`: 上界（不包含）
- `step`: 步长值（必须为正）
- `iter_args`: 循环携带变量的初始值（可选）

**结果：**
- 循环携带变量的最终值（如果有）

**示例：**
```mlir
// 简单循环
scf.for %i = %c0 to %c100 step %c1 {
  // 循环体
}

// 带累加器的循环
%sum = scf.for %i = %c0 to %c100 step %c1 
    iter_args(%acc = %c0_i32) -> (i32) {
  %val = memref.load %array[%i] : memref<?xi32>
  %new_acc = arith.addi %acc, %val : i32
  scf.yield %new_acc : i32
}

// 无符号比较
scf.for unsigned %i = %lb to %ub step %step : i32 {
  // 循环体
}
```

---

### scf.while - While 循环

**描述：** While 循环，在"before"区域中检查条件，在"after"区域中执行循环体。

**语法：**
```mlir
%result = scf.while (%arg = %init) : (type) -> type {
  // before 区域：条件检查
  %condition = ...
  scf.condition(%condition) %arg : type
} do {
^bb0(%arg: type):
  // after 区域：循环体
  %next = ...
  scf.yield %next : type
}
```

**区域：**
- `before`: 条件检查区域（以 `scf.condition` 终止）
- `after`: 循环体区域（以 `scf.yield` 终止）

**示例：**
```mlir
%result = scf.while (%arg = %init) : (i32) -> i32 {
  %condition = arith.cmpi slt, %arg, %limit : i32
  scf.condition(%condition) %arg : i32
} do {
^bb0(%arg: i32):
  %next = arith.addi %arg, %c1 : i32
  scf.yield %next : i32
}
```

---

## 条件操作

### scf.if - If-Then-Else

**描述：** 条件分支，可选 else 块和可选结果。

**语法：**
```mlir
scf.if %condition {
  // then 块
}

scf.if %condition {
  // then 块
} else {
  // else 块
}

%result = scf.if %condition -> type {
  // then 块
  scf.yield %value : type
} else {
  // else 块
  scf.yield %other : type
}
```

**操作数：**
- `condition`: 布尔条件（i1）

**结果：**
- 从分支产生的值（如果有）

**示例：**
```mlir
// 简单 if
scf.if %condition {
  // then 块
}

// If-else
scf.if %condition {
  // then 块
} else {
  // else 块
}

// 带结果的 if
%result = scf.if %condition -> i32 {
  %value = arith.constant 1 : i32
  scf.yield %value : i32
} else {
  %value = arith.constant 0 : i32
  scf.yield %value : i32
}
```

---

### scf.index_switch - 索引切换

**描述：** 基于索引值的 switch 语句，具有多个 case 和一个 default case。

**语法：**
```mlir
%result = scf.index_switch %arg -> type
  case 0 {
    // case 0 块
    scf.yield %value0 : type
  }
  case 1 {
    // case 1 块
    scf.yield %value1 : type
  }
  default {
    // default 块
    scf.yield %default_value : type
  }
```

**操作数：**
- `arg`: 要切换的索引值

**属性：**
- `cases`: case 值数组

**示例：**
```mlir
%result = scf.index_switch %idx -> i32
  case 0 {
    %c0 = arith.constant 10 : i32
    scf.yield %c0 : i32
  }
  case 1 {
    %c1 = arith.constant 20 : i32
    scf.yield %c1 : i32
  }
  case 2 {
    %c2 = arith.constant 30 : i32
    scf.yield %c2 : i32
  }
  default {
    %cd = arith.constant 0 : i32
    scf.yield %cd : i32
  }
```

---

## 区域操作

### scf.execute_region - 执行区域

**描述：** 恰好执行一次区域。允许在单块上下文中使用多个块。

**语法：**
```mlir
%result = scf.execute_region -> type {
  // 区域体（可以有多个块）
  scf.yield %value : type
}

%result = scf.execute_region -> type no_inline {
  // 带 no_inline 属性的区域体
  scf.yield %value : type
}
```

**属性：**
- `no_inline`: 可选标志，防止内联

**语义：**
恰好执行区域一次。用于在通常只允许单个块的操作中创建多块区域。

**示例：**
```mlir
// 简单执行区域
%result = scf.execute_region -> i32 {
  %x = arith.constant 42 : i32
  scf.yield %x : i32
}

// 带多个块
%result = scf.execute_region -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  cf.br ^bb3(%c1 : i32)
^bb2:
  %c2 = arith.constant 2 : i32
  cf.br ^bb3(%c2 : i32)
^bb3(%arg: i32):
  scf.yield %arg : i32
}

// 带 no_inline 属性
%result = scf.execute_region -> i32 no_inline {
  %x = arith.constant 42 : i32
  scf.yield %x : i32
}
```

---

## 终止符操作

### scf.yield - 产生值

**描述：** 终止 SCF 操作内的区域并将值产生给父操作。

**语法：**
```mlir
scf.yield
scf.yield %value : type
scf.yield %value1, %value2 : type1, type2
```

**语义：**
用于终止：

- 循环体（`scf.for`、`scf.while` after 区域）

- 条件分支（`scf.if`）

- 执行区域（`scf.execute_region`）

- Switch case（`scf.index_switch`）

**示例：**
```mlir
// 产生单个值
scf.yield %value : i32

// 产生多个值
scf.yield %a, %b : i32, f32

// 不产生值
scf.yield
```

---

### scf.condition - 循环继续条件

**描述：** 终止 `scf.while` 的"before"区域。如果条件为真，继续到"after"区域；否则退出循环。

**语法：**
```mlir
scf.condition(%condition) %args... : types...
```

**操作数：**
- `condition`: 布尔条件（i1）
- `args`: 要传递给"after"区域或从循环返回的值

**语义：**
- 如果 `condition` 为真：使用 `args` 执行"after"区域
- 如果 `condition` 为假：退出循环并返回 `args`

**示例：**
```mlir
// 在 scf.while before 区域中
%keep_going = arith.cmpi slt, %i, %limit : i32
scf.condition(%keep_going) %i : i32

// 带多个值
%cond = arith.cmpi slt, %i, %limit : i32
scf.condition(%cond) %i, %sum : i32, i32
```

---

## 常见模式

### 模式 1：简单循环
```mlir
scf.for %i = %c0 to %c100 step %c1 {
  // 循环体
}
```

### 模式 2：带累加器的循环
```mlir
%sum = scf.for %i = %c0 to %c100 step %c1 
    iter_args(%acc = %c0) -> (i32) {
  %val = memref.load %array[%i] : memref<?xi32>
  %new_acc = arith.addi %acc, %val : i32
  scf.yield %new_acc : i32
}
```

### 模式 3：带结果的条件
```mlir
%result = scf.if %cond -> i32 {
  scf.yield %true_val : i32
} else {
  scf.yield %false_val : i32
}
```

### 模式 4：While 循环
```mlir
%final = scf.while (%arg = %init) : (i32) -> i32 {
  %keep_going = arith.cmpi slt, %arg, %limit : i32
  scf.condition(%keep_going) %arg : i32
} do {
^bb0(%arg: i32):
  %next = arith.addi %arg, %c1 : i32
  scf.yield %next : i32
}
```

### 模式 5：嵌套循环
```mlir
scf.for %i = %c0 to %M step %c1 {
  scf.for %j = %c0 to %N step %c1 {
    // 嵌套循环体
  }
}
```

### 模式 6：带多个累加器的循环
```mlir
%sum, %prod = scf.for %i = %c0 to %c100 step %c1 
    iter_args(%acc_sum = %c0, %acc_prod = %c1) -> (i32, i32) {
  %val = memref.load %array[%i] : memref<?xi32>
  %new_sum = arith.addi %acc_sum, %val : i32
  %new_prod = arith.muli %acc_prod, %val : i32
  scf.yield %new_sum, %new_prod : i32, i32
}
```


