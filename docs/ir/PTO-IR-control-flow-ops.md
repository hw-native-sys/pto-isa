# Control Flow Operations

This document describes structured control flow operations from the MLIR `scf` (Structured Control Flow) dialect.

**Total Operations:** 9

---

## Loop Operations

### scf.for - For Loop

**Description:** For loop with lower bound, upper bound, and step. Supports loop-carried variables and signed/unsigned comparison.

**Syntax:**
```mlir
scf.for %iv = %lb to %ub step %step {
  // loop body
}

scf.for %iv = %lb to %ub step %step 
    iter_args(%arg = %init) -> (type) {
  // loop body with loop-carried variable
  scf.yield %new_value : type
}

scf.for unsigned %iv = %lb to %ub step %step : i32 {
  // unsigned comparison
}
```

**Operands:**
- `lb`: Lower bound (index or integer)
- `ub`: Upper bound (exclusive)
- `step`: Step value (must be positive)
- `iter_args`: Initial values for loop-carried variables (optional)

**Results:**
- Final values of loop-carried variables (if any)

**Example:**
```mlir
// Simple loop
scf.for %i = %c0 to %c100 step %c1 {
  // loop body
}

// Loop with accumulator
%sum = scf.for %i = %c0 to %c100 step %c1 
    iter_args(%acc = %c0_i32) -> (i32) {
  %val = memref.load %array[%i] : memref<?xi32>
  %new_acc = arith.addi %acc, %val : i32
  scf.yield %new_acc : i32
}

// Unsigned comparison
scf.for unsigned %i = %lb to %ub step %step : i32 {
  // loop body
}
```

---

### scf.while - While Loop

**Description:** While loop with condition check in "before" region and loop body in "after" region.

**Syntax:**
```mlir
%result = scf.while (%arg = %init) : (type) -> type {
  // before region: condition check
  %condition = ...
  scf.condition(%condition) %arg : type
} do {
^bb0(%arg: type):
  // after region: loop body
  %next = ...
  scf.yield %next : type
}
```

**Regions:**
- `before`: Condition check region (terminates with `scf.condition`)
- `after`: Loop body region (terminates with `scf.yield`)

**Example:**
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

## Conditional Operations

### scf.if - If-Then-Else

**Description:** Conditional branch with optional else block and optional results.

**Syntax:**
```mlir
scf.if %condition {
  // then block
}

scf.if %condition {
  // then block
} else {
  // else block
}

%result = scf.if %condition -> type {
  // then block
  scf.yield %value : type
} else {
  // else block
  scf.yield %other : type
}
```

**Operands:**
- `condition`: Boolean condition (i1)

**Results:**
- Values yielded from branches (if any)

**Example:**
```mlir
// Simple if
scf.if %condition {
  // then block
}

// If-else
scf.if %condition {
  // then block
} else {
  // else block
}

// If with results
%result = scf.if %condition -> i32 {
  %value = arith.constant 1 : i32
  scf.yield %value : i32
} else {
  %value = arith.constant 0 : i32
  scf.yield %value : i32
}
```

---

### scf.index_switch - Index Switch

**Description:** Switch statement based on an index value with multiple cases and a default case.

**Syntax:**
```mlir
%result = scf.index_switch %arg -> type
  case 0 {
    // case 0 block
    scf.yield %value0 : type
  }
  case 1 {
    // case 1 block
    scf.yield %value1 : type
  }
  default {
    // default block
    scf.yield %default_value : type
  }
```

**Operands:**
- `arg`: Index value to switch on

**Attributes:**
- `cases`: Array of case values

**Example:**
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

## Region Operations

### scf.execute_region - Execute Region

**Description:** Execute a region exactly once. Allows multiple blocks within single-block contexts.

**Syntax:**
```mlir
%result = scf.execute_region -> type {
  // region body (can have multiple blocks)
  scf.yield %value : type
}

%result = scf.execute_region -> type no_inline {
  // region body with no_inline attribute
  scf.yield %value : type
}
```

**Attributes:**
- `no_inline`: Optional flag to prevent inlining

**Semantics:**
Executes the region exactly once. Useful for creating multi-block regions within operations that normally allow only single blocks.

**Example:**
```mlir
// Simple execute region
%result = scf.execute_region -> i32 {
  %x = arith.constant 42 : i32
  scf.yield %x : i32
}

// With multiple blocks
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

// With no_inline attribute
%result = scf.execute_region -> i32 no_inline {
  %x = arith.constant 42 : i32
  scf.yield %x : i32
}
```

---

## Terminator Operations

### scf.yield - Yield Values

**Description:** Terminates regions within SCF operations and yields values to parent operation.

**Syntax:**
```mlir
scf.yield
scf.yield %value : type
scf.yield %value1, %value2 : type1, type2
```

**Semantics:**
Used to terminate:

- Loop bodies (`scf.for`, `scf.while` after region)

- Conditional branches (`scf.if`)

- Execute regions (`scf.execute_region`)

- Switch cases (`scf.index_switch`)

**Example:**
```mlir
// Yield single value
scf.yield %value : i32

// Yield multiple values
scf.yield %a, %b : i32, f32

// Yield no values
scf.yield
```

---

### scf.condition - Loop Continuation Condition

**Description:** Terminates the "before" region of `scf.while`. If condition is true, continues to "after" region; otherwise exits loop.

**Syntax:**
```mlir
scf.condition(%condition) %args... : types...
```

**Operands:**
- `condition`: Boolean condition (i1)
- `args`: Values to pass to "after" region or return from loop

**Semantics:**
- If `condition` is true: execute "after" region with `args`
- If `condition` is false: exit loop and return `args`

**Example:**
```mlir
// In scf.while before region
%keep_going = arith.cmpi slt, %i, %limit : i32
scf.condition(%keep_going) %i : i32

// With multiple values
%cond = arith.cmpi slt, %i, %limit : i32
scf.condition(%cond) %i, %sum : i32, i32
```

---

## Common Patterns

### Pattern 1: Simple Loop
```mlir
scf.for %i = %c0 to %c100 step %c1 {
  // loop body
}
```

### Pattern 2: Loop with Accumulator
```mlir
%sum = scf.for %i = %c0 to %c100 step %c1 
    iter_args(%acc = %c0) -> (i32) {
  %val = memref.load %array[%i] : memref<?xi32>
  %new_acc = arith.addi %acc, %val : i32
  scf.yield %new_acc : i32
}
```

### Pattern 3: Conditional with Results
```mlir
%result = scf.if %cond -> i32 {
  scf.yield %true_val : i32
} else {
  scf.yield %false_val : i32
}
```

### Pattern 4: While Loop
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

### Pattern 5: Nested Loops
```mlir
scf.for %i = %c0 to %M step %c1 {
  scf.for %j = %c0 to %N step %c1 {
    // nested loop body
  }
}
```

### Pattern 6: Loop with Multiple Accumulators
```mlir
%sum, %prod = scf.for %i = %c0 to %c100 step %c1 
    iter_args(%acc_sum = %c0, %acc_prod = %c1) -> (i32, i32) {
  %val = memref.load %array[%i] : memref<?xi32>
  %new_sum = arith.addi %acc_sum, %val : i32
  %new_prod = arith.muli %acc_prod, %val : i32
  scf.yield %new_sum, %new_prod : i32, i32
}
```

---

