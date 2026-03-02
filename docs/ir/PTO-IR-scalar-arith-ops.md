# Scalar Arithmetic Operations

This document describes scalar arithmetic operations from the MLIR `arith` dialect. 

**Important:** PTO AS only supports **scalar operations** from the arith dialect. Vector and tensor operations are not supported.

**Total Operations:** 47

**Note:** The operations `arith.scaling_extf` and `arith.scaling_truncf` are not included as they are not supported in PTO AS.

---

## Integer Arithmetic Operations

### arith.addi - Integer Addition

**Description:** Integer addition with optional overflow flags.

**Syntax:**
```mlir
%result = arith.addi %lhs, %rhs : i32
%result = arith.addi %lhs, %rhs overflow<nsw, nuw> : i32
```

**Example:**
```mlir
// Scalar addition
%a = arith.addi %b, %c : i64
```

---

### arith.subi - Integer Subtraction

**Syntax:**
```mlir
%result = arith.subi %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar subtraction
%a = arith.subi %b, %c : i32
```

---

### arith.muli - Integer Multiplication

**Syntax:**
```mlir
%result = arith.muli %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar multiplication
%a = arith.muli %b, %c : i64
```

---

### arith.divsi - Signed Integer Division

**Description:** Signed integer division, rounds towards zero.

**Syntax:**
```mlir
%result = arith.divsi %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar signed division
%a = arith.divsi %b, %c : i32
```

---

### arith.divui - Unsigned Integer Division

**Syntax:**
```mlir
%result = arith.divui %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar unsigned division
%a = arith.divui %b, %c : i32
```

---

### arith.remsi - Signed Integer Remainder

**Syntax:**
```mlir
%result = arith.remsi %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar signed remainder
%a = arith.remsi %b, %c : i32
```

---

### arith.remui - Unsigned Integer Remainder

**Syntax:**
```mlir
%result = arith.remui %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar unsigned remainder
%a = arith.remui %b, %c : i32
```

---

### arith.ceildivsi - Ceiling Division (Signed)

**Description:** Signed integer division rounding towards positive infinity.

**Syntax:**
```mlir
%result = arith.ceildivsi %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar ceiling division
%a = arith.ceildivsi %b, %c : i32
```

---

### arith.ceildivui - Ceiling Division (Unsigned)

**Syntax:**
```mlir
%result = arith.ceildivui %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar ceiling division (unsigned)
%a = arith.ceildivui %b, %c : i32
```

---

### arith.floordivsi - Floor Division (Signed)

**Description:** Signed integer division rounding towards negative infinity.

**Syntax:**
```mlir
%result = arith.floordivsi %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar floor division
%a = arith.floordivsi %b, %c : i32
```

---

## Floating-Point Arithmetic Operations

### arith.addf - Floating-Point Addition

**Syntax:**
```mlir
%result = arith.addf %lhs, %rhs : f32
%result = arith.addf %lhs, %rhs fastmath<fast> : f32
```

**Example:**
```mlir
// Scalar addition
%a = arith.addf %b, %c : f64
```

---

### arith.subf - Floating-Point Subtraction

**Syntax:**
```mlir
%result = arith.subf %lhs, %rhs : f32
```

**Example:**
```mlir
// Scalar subtraction
%a = arith.subf %b, %c : f32
```

---

### arith.mulf - Floating-Point Multiplication

**Syntax:**
```mlir
%result = arith.mulf %lhs, %rhs : f32
```

**Example:**
```mlir
// Scalar multiplication
%a = arith.mulf %b, %c : f64
```

---

### arith.divf - Floating-Point Division

**Syntax:**
```mlir
%result = arith.divf %lhs, %rhs : f32
```

**Example:**
```mlir
// Scalar division
%a = arith.divf %b, %c : f32
```

---

### arith.remf - Floating-Point Remainder

**Syntax:**
```mlir
%result = arith.remf %lhs, %rhs : f32
```

**Example:**
```mlir
// Scalar remainder
%a = arith.remf %b, %c : f64
```

---

### arith.negf - Floating-Point Negation

**Syntax:**
```mlir
%result = arith.negf %operand : f32
```

**Example:**
```mlir
// Scalar negation
%a = arith.negf %b : f32
```

---

## Bitwise Operations

### arith.andi - Bitwise AND

**Syntax:**
```mlir
%result = arith.andi %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar bitwise AND
%a = arith.andi %b, %c : i32
```

---

### arith.ori - Bitwise OR

**Syntax:**
```mlir
%result = arith.ori %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar bitwise OR
%a = arith.ori %b, %c : i64
```

---

### arith.xori - Bitwise XOR

**Syntax:**
```mlir
%result = arith.xori %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar bitwise XOR
%a = arith.xori %b, %c : i32
```

---

## Shift Operations

### arith.shli - Shift Left

**Syntax:**
```mlir
%result = arith.shli %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar shift left
%a = arith.shli %b, %c : i32
```

---

### arith.shrsi - Arithmetic Shift Right (Signed)

**Description:** Arithmetic right shift (sign-extended).

**Syntax:**
```mlir
%result = arith.shrsi %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar arithmetic shift right
%a = arith.shrsi %b, %c : i32
```

---

### arith.shrui - Logical Shift Right (Unsigned)

**Description:** Logical right shift (zero-extended).

**Syntax:**
```mlir
%result = arith.shrui %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar logical shift right
%a = arith.shrui %b, %c : i32
```

---

## Comparison Operations

### arith.cmpi - Integer Comparison

**Description:** Compare two integers with specified predicate.

**Syntax:**
```mlir
%result = arith.cmpi <predicate>, %lhs, %rhs : i32
```

**Predicates:**
- Signed: `slt`, `sle`, `sgt`, `sge`
- Unsigned: `ult`, `ule`, `ugt`, `uge`
- Equality: `eq`, `ne`

**Example:**
```mlir
// Scalar comparison
%cmp = arith.cmpi slt, %a, %b : i32
%eq = arith.cmpi eq, %x, %y : i64
```

---

### arith.cmpf - Floating-Point Comparison

**Description:** Compare two floats with specified predicate.

**Syntax:**
```mlir
%result = arith.cmpf <predicate>, %lhs, %rhs : f32
```

**Predicates:**
- Ordered: `oeq`, `one`, `olt`, `ole`, `ogt`, `oge`, `ord`
- Unordered: `ueq`, `une`, `ult`, `ule`, `ugt`, `uge`, `uno`
- Always: `true`, `false`

**Example:**
```mlir
// Scalar comparison
%cmp = arith.cmpf olt, %a, %b : f32
%eq = arith.cmpf oeq, %x, %y : f64
```

---

## Min/Max Operations

### arith.minsi - Minimum (Signed Integer)

**Syntax:**
```mlir
%result = arith.minsi %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar minimum
%min = arith.minsi %a, %b : i32
```

---

### arith.minui - Minimum (Unsigned Integer)

**Syntax:**
```mlir
%result = arith.minui %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar minimum (unsigned)
%min = arith.minui %a, %b : i32
```

---

### arith.maxsi - Maximum (Signed Integer)

**Syntax:**
```mlir
%result = arith.maxsi %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar maximum
%max = arith.maxsi %a, %b : i32
```

---

### arith.maxui - Maximum (Unsigned Integer)

**Syntax:**
```mlir
%result = arith.maxui %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar maximum (unsigned)
%max = arith.maxui %a, %b : i32
```

---

### arith.minimumf - Minimum (Float, propagates NaN)

**Syntax:**
```mlir
%result = arith.minimumf %lhs, %rhs : f32
```

**Example:**
```mlir
// Scalar minimum (propagates NaN)
%min = arith.minimumf %a, %b : f32
```

---

### arith.maximumf - Maximum (Float, propagates NaN)

**Syntax:**
```mlir
%result = arith.maximumf %lhs, %rhs : f32
```

**Example:**
```mlir
// Scalar maximum (propagates NaN)
%max = arith.maximumf %a, %b : f32
```

---

### arith.minnumf - Minimum (Float, ignores NaN)

**Syntax:**
```mlir
%result = arith.minnumf %lhs, %rhs : f32
```

**Example:**
```mlir
// Scalar minimum (ignores NaN)
%min = arith.minnumf %a, %b : f64
```

---

### arith.maxnumf - Maximum (Float, ignores NaN)

**Syntax:**
```mlir
%result = arith.maxnumf %lhs, %rhs : f32
```

**Example:**
```mlir
// Scalar maximum (ignores NaN)
%max = arith.maxnumf %a, %b : f64
```

---

## Type Conversion Operations

### arith.extsi - Sign Extension

**Description:** Sign-extend integer to wider type.

**Syntax:**
```mlir
%result = arith.extsi %in : i32 to i64
```

**Example:**
```mlir
// Scalar sign extension
%wide = arith.extsi %narrow : i32 to i64
```

---

### arith.extui - Zero Extension

**Description:** Zero-extend integer to wider type.

**Syntax:**
```mlir
%result = arith.extui %in : i32 to i64
```

**Example:**
```mlir
// Scalar zero extension
%wide = arith.extui %narrow : i32 to i64
```

---

### arith.trunci - Truncate Integer

**Description:** Truncate integer to narrower type.

**Syntax:**
```mlir
%result = arith.trunci %in : i64 to i32
```

**Example:**
```mlir
// Scalar truncation
%narrow = arith.trunci %wide : i64 to i32
```

---

### arith.extf - Extend Float

**Description:** Extend floating-point to wider type.

**Syntax:**
```mlir
%result = arith.extf %in : f32 to f64
```

**Example:**
```mlir
// Scalar float extension
%double = arith.extf %single : f32 to f64
```

---

### arith.truncf - Truncate Float

**Description:** Truncate floating-point to narrower type.

**Syntax:**
```mlir
%result = arith.truncf %in : f64 to f32
```

**Example:**
```mlir
// Scalar float truncation
%single = arith.truncf %double : f64 to f32
```

---

### arith.sitofp - Signed Integer to Float

**Syntax:**
```mlir
%result = arith.sitofp %in : i32 to f32
```

**Example:**
```mlir
// Scalar int to float
%fp = arith.sitofp %int : i32 to f32
```

---

### arith.uitofp - Unsigned Integer to Float

**Syntax:**
```mlir
%result = arith.uitofp %in : i32 to f32
```

**Example:**
```mlir
// Scalar uint to float
%fp = arith.uitofp %uint : i32 to f32
```

---

### arith.fptosi - Float to Signed Integer

**Description:** Convert float to signed integer (rounds towards zero).

**Syntax:**
```mlir
%result = arith.fptosi %in : f32 to i32
```

**Example:**
```mlir
// Scalar float to int
%int = arith.fptosi %fp : f32 to i32
```

---

### arith.fptoui - Float to Unsigned Integer

**Syntax:**
```mlir
%result = arith.fptoui %in : f32 to i32
```

**Example:**
```mlir
// Scalar float to uint
%uint = arith.fptoui %fp : f32 to i32
```

---

### arith.bitcast - Bitcast

**Description:** Reinterpret bits as different type (same bit width).

**Syntax:**
```mlir
%result = arith.bitcast %in : f32 to i32
```

**Example:**
```mlir
// Scalar bitcast
%bits = arith.bitcast %fp : f32 to i32
```

---

### arith.index_cast - Index Cast (Signed)

**Description:** Cast between `index` type and integer types (sign-extended).

**Syntax:**
```mlir
%result = arith.index_cast %in : i32 to index
%result = arith.index_cast %in : index to i64
```

**Example:**
```mlir
// Scalar index cast
%idx = arith.index_cast %int : i32 to index
%int = arith.index_cast %idx : index to i64
```

---

### arith.index_castui - Index Cast (Unsigned)

**Description:** Cast between `index` type and integer types (zero-extended).

**Syntax:**
```mlir
%result = arith.index_castui %in : i32 to index
```

**Example:**
```mlir
// Scalar index cast (unsigned)
%idx = arith.index_castui %uint : i32 to index
```

---

## Special Operations

### arith.select - Conditional Select

**Syntax:**
```mlir
%result = arith.select %condition, %true_value, %false_value : i32
```

**Example:**
```mlir
// Scalar select
%result = arith.select %cond, %a, %b : i32
%fp_result = arith.select %cond, %x, %y : f32
```

---

### arith.constant - Constant Value

**Syntax:**
```mlir
%result = arith.constant <value> : <type>
```

**Example:**
```mlir
// Scalar constants
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i64
%pi = arith.constant 3.14159 : f32
%true = arith.constant true
```

---

## Extended Arithmetic Operations

### arith.addui_extended - Extended Unsigned Addition

**Description:** Unsigned addition with overflow flag.

**Syntax:**
```mlir
%sum, %overflow = arith.addui_extended %lhs, %rhs : i32, i1
```

**Example:**
```mlir
// Scalar extended addition
%sum, %overflow = arith.addui_extended %a, %b : i32, i1
```

---

### arith.mulsi_extended - Extended Signed Multiplication

**Description:** Signed multiplication returning low and high bits.

**Syntax:**
```mlir
%low, %high = arith.mulsi_extended %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar extended multiplication
%low, %high = arith.mulsi_extended %a, %b : i32
```

---

### arith.mului_extended - Extended Unsigned Multiplication

**Syntax:**
```mlir
%low, %high = arith.mului_extended %lhs, %rhs : i32
```

**Example:**
```mlir
// Scalar extended multiplication (unsigned)
%low, %high = arith.mului_extended %a, %b : i32
```

---




