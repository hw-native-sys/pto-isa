# Type System

PTO uses a compact visible type system, but legality does not stop at raw type names. Type classes tell you what kind of architectural object you are dealing with. Other legality dimensions such as layout, location, valid region, and target profile determine whether a use is actually allowed.

## Element Types

PTO supports a rich set of element types across floating-point, integer, and specialized categories.

### Floating-Point Types

| Type | SSA Name | Bits | Description | A2/A3 | A5 |
|------|----------|------|-------------|:------:|:--:|
| IEEE FP16 | `f16` / `half` | 16 | IEEE 754 half-precision | Yes | Yes |
| BF16 | `bf16` / `bfloat16_t` | 16 | Brain float 16 (8-bit exponent) | Yes | Yes |
| IEEE FP32 | `f32` | 32 | IEEE 754 single-precision | Yes | Yes |
| FP8 E4M3 | `f8e4m3` / `float8_e4m3_t` | 8 | 4-bit exponent, 3-bit mantissa | No | Yes |
| FP8 E5M2 | `f8e5m2` / `float8_e5m2_t` | 8 | 5-bit exponent, 2-bit mantissa | No | Yes |
| HI Float8 | `hifloat8_t` | 8 | High-precision float8 | No | Yes |
| Float4 E1M2x2 | `float4_e1m2x2_t` | 4 | 4-bit float4, packed 2x2 | No | Yes |
| Float4 E2M1x2 | `float4_e2m1x2_t` | 4 | 4-bit float4, packed 2x2 | No | Yes |

### Integer Types

| Type | SSA Name | Bits | Signedness | A2/A3 | A5 |
|------|----------|------|------------|:------:|:--:|
| int8 | `i8` | 8 | Signed | Yes | Yes |
| uint8 | `u8` | 8 | Unsigned | Yes | Yes |
| int16 | `i16` | 16 | Signed | Yes | Yes |
| uint16 | `u16` | 16 | Unsigned | Yes | Yes |
| int32 | `i32` | 32 | Signed | Yes | Yes |
| uint32 | `u32` | 32 | Unsigned | Yes | Yes |
| int64 | `i64` | 64 | Signed | Yes | Yes |
| uint64 | `u64` | 64 | Unsigned | Yes | Yes |

## Vector Width

The vector register width `N` (the number of lanes) is determined by the element type and the target profile:

| Element Type | Vector Width N | Bytes/Register | Notes |
|-------------|:-------------:|:-------------:|-------|
| f32 | 64 | 256 B | 64 × 32-bit |
| f16, bf16 | 128 | 256 B | 128 × 16-bit |
| i16, u16 | 128 | 256 B | 128 × 16-bit |
| i8, u8 | 256 | 256 B | 256 × 8-bit |
| f8e4m3, f8e5m2 | 256 | 256 B | 256 × 8-bit |

Vector width is **portable** across all profiles: CPU simulation, A2/A3, and A5 all present the same `N` value for each element type. The difference is that A5 executes vector operations natively on hardware, while CPU/A2/A3 emulate them.

## Vector Register Types

Vector register SSA type: `!pto.vreg<NxDTYPE>`

```
!pto.vreg<64xf32>   -- 64 lanes of f32
!pto.vreg<128xf16>  -- 128 lanes of f16
!pto.vreg<256xi8>   -- 256 lanes of i8
```

## Tile Buffer Types

Tile buffer SSA type (see [Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md) for full parameter list):

```
!pto.tile<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>
!pto.tile_buf<loc=mat, bf16, 16, 16, RowMajor, NoneBox, None, Null>
!pto.tile_buf<loc=left, int8, 16, 16, RowMajor, RowMajor, NZ, Null>
```

## NaN and Inf Behavior

For floating-point types, PTO follows IEEE 754 semantics with the following hardware-specific variation points:

| Behavior | Rule |
|----------|------|
| Quiet NaN propagation | Quiet NaN in → quiet NaN out (preserves signaling bit) |
| Signaling NaN | Signaling NaN may be quieted by hardware before use |
| Inf arithmetic | Inf is produced and propagated as IEEE 754 requires |
| Denormalized numbers | Hardware may flush denormals to zero (FTZ behavior) |
| Rounding | Controlled by `rnd` attribute: `rne` (default), `rz`, `rp`, `rm` |

The FTZ (flush-to-zero) behavior for denormals is **hardware-specific**:

- **A2/A3**: Denormals are flushed to zero by default. Hardware does not generate subnormal results during arithmetic; operands that are denormal are treated as zero.
- **A5**: Denormals are flushed to zero by default. The `settf32mode`/`settf32mode` instructions can modify this behavior on A5-class targets.
- **CPU simulator**: Denormals may be preserved or flushed depending on the host CPU's SSE/AVX floating-point configuration; behavior is not guaranteed to match the hardware targets.

## Type Conversion Rules

### Between Floating-Point Types

| Source | Dest | Behavior |
|--------|------|----------|
| f16 → bf16 | Conversion | Reinterpret f16 bits as bf16 (no numerical conversion) |
| bf16 → f16 | Conversion | Reinterpret bf16 bits as f16 (no numerical conversion) |
| f16/bf16 → f32 | Promotion | Extend to f32; exact representable values are preserved |
| f32 → f16/bf16 | Narrowing | Round according to `rnd` attribute; NaN/Inf handled per IEEE 754 |
| f8 → f16/f32 | Promotion | Extend; exact representable values are preserved |
| f16/f32 → f8 | Narrowing | Round according to `rnd` attribute; may overflow to Inf |

### Between Integer Types

| Source | Dest | Behavior |
|--------|------|----------|
| Widening (e.g., i8 → i16) | Zero/_sign extend | Zero-extend for unsigned; sign-extend for signed |
| Narrowing (e.g., i16 → i8) | Truncation | Truncate high bits; may lose significant bits |
| i32 → f32 | Conversion | Exact for values in [-2^24, 2^24]; may lose precision outside |
| f32 → i32 | Conversion | Truncates toward zero; on overflow, result is undefined behavior on A2/A3 and A5 (saturates on CPU simulator) |

### Between Float and Integer

| Source | Dest | Behavior |
|--------|------|----------|
| f32 → i8/u8/i16/u16 | Narrowing | Truncate; may overflow |
| f32 → i32/u32 | Narrowing | Truncate; may overflow |
| i8/u8 → f32 | Promotion | Exact for small values; may lose precision for large values |

### Type Conversion Operations

| Operation | Instruction Set | Description |
|-----------|---------|-------------|
| `pto.tcvt` | Tile | Elementwise type conversion on tile buffers |
| `pto.vcvt` | Vector | Vector register type conversion |
| `pto.vtrc` | Vector | Vector truncate/round (e.g., f32 → f16) |
| `pto.vci` | Vector | Compress to integer (vector → integer result) |

## Constraints

- Instruction set pages must define accepted operand/result classes.
- Type errors must stay distinguishable from deeper legality failures (shape, layout, location intent, target profile).
- Vector instruction docs must make vector-register, mask, pointer, and alignment state explicit.
- Tile instruction docs must make tile role, shape, and valid-region interactions explicit.
- No implicit type promotion: `tadd(t, i8_tile, f32_immediate)` is illegal unless an explicit `tcvt` converts one operand first.

## Cases That Are Not Allowed

- Treating type class checks as though they cover every backend legality fact.
- Conflating scalar state with tile or vector payload state.
- Documenting vector and tile payload classes as if they were interchangeable.
- Relying on implicit type conversion without an explicit `tcvt`/`vcvt`.

## See Also

- [Location Intent And Legality](./location-intent-and-legality.md)
- [Instruction Sets](../instruction-families/README.md)
- [Source Of Truth](../reference/source-of-truth.md)
- [Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md)
