# TFILLPAD Test Cases and API Usage

## Overview

TFILLPAD (Tile Fill Pad) is a PTO instruction that copies data from a source tile to a destination tile while padding the destination tile to its full dimensions. It supports various data types and padding value options.

## Test Cases

| Case | Data Type | Global Shape | Tile Shape | Pad Row | Pad Col | Description |
|------|-----------|--------------|------------|---------|---------|-------------|
| 1 | float | 128×127 | 128×128 | Max | Max | Basic float padding |
| 2 | float | 128×127 | 128×160 | Max | Max | Wider tile padding |
| 3 | float | 128×127 | 128×160 | Min | Max | Mixed pad values |
| 4 | float | 260×7 | 260×16 | Min | Max | Large row count, small cols |
| 5 | float | 260×7 | 260×16 | Min | Max | Inplace variant (same src/dst) |
| 6 | uint16 | 260×7 | 260×32 | Min | Max | 16-bit unsigned integer |
| 7 | int8 | 260×7 | 260×64 | Min | Max | 8-bit signed integer |
| 8 | uint16 | 259×7 | 260×32 | Min | Max | Row expansion (259→260) |
| 9 | int8 | 259×7 | 260×64 | Min | Max | Row expansion with int8 |
| 10 | float | 128×64 | 128×128 | Null | Custom(-1.0f) | **Custom pad value** |

## Standard Pad Values

| PadValue | Description | float | int32 | int16 | int8 |
|----------|-------------|-------|-------|-------|------|
| `Null` | No padding (0) | 0 | 0 | 0 | 0 |
| `Zero` | Zero fill | 0 | 0 | 0 | 0 |
| `Min` | Minimum value | -∞ | INT32_MIN | INT16_MIN | INT8_MIN |
| `Max` | Maximum value | +∞ | INT32_MAX | INT16_MAX | INT8_MAX |

## Custom Pad Values (New API)

Custom pad values allow specifying arbitrary padding values at compile time.

### Using `PadValueCustom()` Function (Recommended)

```cpp
#include <pto/common/constants.hpp>

// Create a custom pad value
constexpr PadValue PadCustomNeg1 = PadValueCustom(-1.0f);
constexpr PadValue PadCustomHalf = PadValueCustom(0.5f);
constexpr PadValue PadCustom42   = PadValueCustom(42.0f);

// Use in TileDyn declaration
using MyTile = TileDyn<float, 128, 128, TileType::Vec, PadCustomNeg1>;
```

### Using `PadCustom<V>` Template (Alternative)

```cpp
// Template variable syntax
constexpr PadValue MyPad = PadCustom<-1.0f>;
```

### How It Works

Custom pad values encode the float's bit pattern into the upper 32 bits of a 64-bit enum:

```cpp
// Internal representation (uint64_t):
// [63:32] = float bit pattern
// [31:0]  = CustomBase marker (0x100000000)

// Example: -1.0f has bit pattern 0xBF800000
// Encoded as: 0xBF80000000000001ULL
```

The `GetPadValue<TileData>()` function (used by A2A3/A5/CPU implementations) automatically detects and decodes custom values:

```cpp
if constexpr (isCustomPadValue(PadVal)) {
    constexpr uint32_t bits = getCustomPadBits(PadVal);
    // Decode based on DType size
}
```

## API Functions

### `PadValueCustom(float value)` → `PadValue`

Creates a custom PadValue from a float. Constexpr-compatible.

```cpp
constexpr PadValue PadValueCustom(float value);
```

### `isCustomPadValue(PadValue pv)` → `bool`

Returns true if the PadValue is a custom value (not Null/Zero/Min/Max).

```cpp
constexpr bool isCustomPadValue(PadValue pv);
```

### `getCustomPadBits(PadValue pv)` → `uint32_t`

Extracts the raw bit pattern from a custom PadValue.

```cpp
constexpr uint32_t getCustomPadBits(PadValue pv);
```

## Platform Support

| Platform | Standard Values | Custom Values |
|----------|-----------------|---------------|
| CPU Sim  | ✅ | ✅ |
| A2/A3    | ✅ | ✅ (via GetPadValue) |
| A5       | ✅ | ✅ (via GetPadValue) |

## Example: TFILLPAD with Custom Pad Value

```cpp
#include <pto/pto-inst.hpp>

// Define custom pad value
constexpr PadValue PadCustomNeg1 = PadValueCustom(-1.0f);

// Source tile (128x64 valid data)
using SrcTile = TileDyn<float, 128, 64, TileType::Vec, PadValue::Null>;

// Destination tile (128x128, pad with -1.0f)
using DstTile = TileDyn<float, 128, 128, TileType::Vec, PadCustomNeg1>;

void example() {
    SrcTile src(128, 64);
    DstTile dst(128, 128);

    // Fill source with data...

    // TFILLPAD copies src to dst, pads remaining cols with -1.0f
    TFILLPAD(dst, src);

    // Result:
    // - dst[0:128, 0:64]   = src data
    // - dst[0:128, 64:128] = -1.0f (padded)
}
```

## Files Modified for Custom Value Support

1. `include/pto/common/type.hpp` - PadValue enum changed to `uint64_t` underlying type
2. `include/pto/common/constants.hpp` - Added:
   - `isCustomPadValue()`, `getCustomPadBits()`
   - `PadValueCustom()` function
   - `PadCustom<V>` template
   - Updated `GetPadValue()` to handle custom values
3. `include/pto/cpu/TFillPad.hpp` - Custom value handling in CPU sim
4. A2A3/A5 - Use `GetPadValue()` which now handles custom values automatically
