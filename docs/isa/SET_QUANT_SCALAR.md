# SET_QUANT_SCALAR

## Introduction

Set the scalar quantization parameter (pre-quantization scale) for subsequent `TPUSH` operations. The scalar value is encoded into the hardware quantization configuration register based on the output data type.

For 8-bit output types (`int8_t` or `uint8_t`), the sign bit is automatically encoded into the configuration value at bit position 46.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename OutType, typename... WaitEvents>
PTO_INST RecordEvent SET_QUANT_SCALAR(float preQuantScalar, WaitEvents &...events);
```

## Constraints

- `OutType` must be a supported quantization output type (e.g. `int8_t`, `uint8_t`, `int16_t`, `bfloat16_t`, `half`).
- `preQuantScalar` is an FP32 scalar value that specifies the pre-quantization scale factor.
- This instruction must be called before the `TPUSH` instruction that consumes this configuration.
- The exact encoding of the scalar value into the hardware register is implementation-defined.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_set_quant_scalar()
{
    float preQuantScale = 0.5f;
    SET_QUANT_SCALAR<int8_t>(preQuantScale);
}
```

## ASM Form Examples

The current public assembly reference does not define a stable PTO-AS spelling for `SET_QUANT_SCALAR`. Use the C++ intrinsic form for quantization configuration.
