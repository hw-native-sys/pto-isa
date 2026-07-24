# TQUANT HiFloat4 (BF16 → HiF4) — Algorithm & CCE Mapping

## 1. The HiFloat4 Standard (arXiv:2602.11287)

HiFloat4 (HiF4) encodes 64 4-bit floating-point values together with **32 bits** of
three-level shared scaling metadata. The result is 4.5 bits/value of information
density — enough to drive a Cube matmul directly from packed FP4 codes.

### 1.1 The three-level hierarchy

For a group of 64 consecutive elements, HiF4 computes:

| Level | Group size | Metadata    | Bits per element |
|-------|-----------|-------------|------------------|
| Ea    | 64        | 8-bit exponent (e6m2) | 8/64 = 0.125 |
| Eb    | 8         | 1-bit exponent         | 1/8  = 0.125 |
| Ec    | 4         | 1-bit exponent         | 1/4  = 0.25  |

Total metadata: 8 + 8×1 + 16×1 = **32 bits per 64 elements** = 4.5 bits/value.

### 1.2 Data element: FP4 e1m2

Each 4-bit data element uses the **e1m2** format (1 sign, 1 exponent, 2 mantissa):

| Code | Sign | Exp | Mant | Value |
|------|------|-----|------|-------|
| 0000 | 0    | 0   | 00   | +0.0   |
| 0001 | 0    | 0   | 01   | +0.25  |
| 0010 | 0    | 0   | 10   | +0.5   |
| 0011 | 0    | 0   | 11   | +0.75  |
| 0100 | 0    | 1   | 00   | +1.0   |
| 0101 | 0    | 1   | 01   | +1.25  |
| 0110 | 0    | 1   | 10   | +1.5   |
| 0111 | 0    | 1   | 11   | +1.75  |
| 1000 | 1    | 0   | 00   | −0.0   |
| ...  | ...  | ... | ...  | (negatives mirror) |

Bias = 0, implicit leading 0 (not 1) when exp=0 (subnormal-style), implicit
leading 1 when exp=1. The representable range is [0, ±1.75].

### 1.3 The quantization algorithm (paper Algorithm 1)

For each element `x` in a 64-element block:

```
Ma = max(|x|) over the 64-element block          (max per 64)
Mb = max(|x|) over the 8-element sub-block       (max per 8)
Mc = max(|x|) over the 4-element sub-block       (max per 4)

Ea = clamp(floor(log2(Ma)) + offset, 0, 63)      (8-bit, offset = 38)

Eb = 1 if (Mb / 2^Ea) >= 4 else 0               (1-bit per 8-group)

Ec = 1 if (Mc / 2^Ea / 2^Eb) >= 2 else 0        (1-bit per 4-group)

scale = 2^(-Ea) × 2^(-Eb) × 2^(-Ec)

q = round_to_nearest_e1m2(x × scale)
```

The effective shared exponent per element is `Ea + Eb + Ec`, giving a dynamic
range of `2^0 ... 2^63` with 3 bits of fine-grained per-sub-block refinement.


## 2. CCE Implementation Mapping (`a6/TQuant.hpp`)

The CCE pipeline is split into stages, each operating on a VL of 128 BF16
elements (256 bytes) or 256 elements via de-interleave.

### 2.1 Stage 1: AbsReduceMax (gp4 / gp8 / gp64)

Computes `Ma` (per-64), `Mb` (per-8), `Mc` (per-4) from the BF16 source.

```
AbsReduceMax_gp4_Cont:    src[0..511] → Mc[0..127]   (128 per-4 maxima)
AbsReduceMax_gp8_gp64_Cont: Mc[0..127] → Mb[0..63], Ma[0..7]
    Mb = vmax(Mc[i*2], Mc[i*2+1])                    (pairwise max → per-8)
    Ma = vcgmax(Mc)                                   (block reduce → per-64)
```

### 2.2 Stage 2: CalcExpScale_Cont

Reads Ma/Mb/Mc, produces Ea/Eb/Ec exponents + the per-4 scaling factor.

```
Per loop (128 elements = 8 blocks of 16):
  Ma: 8 values (1 per 64-elem block)   — loaded E2B_B16 (broadcast to 16 lanes)
  Mb: 64 values (1 per 8-elem group)   — loaded US_B16 (upscaled, 2× repeat)
  Mc: 128 values (1 per 4-elem group)  — loaded NORM

  // Ea: BF16 → e6m2 conversion (round-to-nearest), then reciprocal back to BF16
  Ea_e6m2 = vcvt_bf162e6m2(Ma, ROUND_R, PART_EVEN)
  Ea_rec  = vcvt_rcpe6m22bf16(Ea_e6m2, PART_EVEN)    // ≈ 2^(-Ea) as BF16

  // Eb: threshold check — is the per-8 max more than 4× the shared exponent?
  Eb_tmp  = Mb * Ea_rec
  Eb_bit  = (Eb_tmp >= 4) ? 1 : 0                     // psts as packed predicate

  // Ec: threshold check — refine further with the Eb correction
  Eb_rec  = Eb_bit ? 0.5 : 1.0                        // 2^(-Eb)
  Ec_tmp  = Mc * Ea_rec * Eb_rec
  Ec_bit  = (Ec_tmp >= 2) ? 1 : 0

  // Final per-4 scaling factor (inverse of the full shared exponent)
  Ec_rec  = Ec_bit ? 0.5 : 1.0                        // 2^(-Ec)
  e_scale = Ea_rec * Eb_rec * Ec_rec                  // ≈ 2^(-(Ea+Eb+Ec))
```

**Threshold semantics:**
- `Mb * Ea_rec >= 4` means `Mb / 2^Ea >= 4`, i.e. the 8-element sub-block max
  needs 2 more exponent bits → `Eb = 1` contributes `2^1 = 2` extra range.
- `Mc * Ea_rec * Eb_rec >= 2` means `Mc / 2^(Ea+Eb) >= 2`, i.e. the 4-element
  sub-block max needs 1 more exponent bit → `Ec = 1`.

### 2.3 Stage 2b: ExpLayoutForCube (CCE-only, NOT in golden)

Rearranges Ea/Eb/Ec from their per-level flat layout into the interleaved
packed layout the Cube matmul unit consumes. This is a hardware-specific
data-movement step — **the Python golden does NOT replicate it.**

### 2.4 Stage 3: CalcFp4Values_Cont

```
Per loop (256 elements):
  input    = vlds(srcPtr, 128*loop, NORM)             // 128 BF16 elements
  e_scale  = vlds(scalingPtr, 64*loop, US_B16)        // per-4 scale, 2× repeat
  scaled   = input * e_scale
  fp4_code = vcvt(scaled, ROUND_A, PART_P0)           // BF16 → f4e1m2x2
  vsts(fp4_code, dstPtr, 64*loop, PK4_B32)            // pack 4 nibbles/byte
```

### 2.5 The e6m2 exponent format

Ea is stored as an 8-bit **e6m2** value (6 exponent bits, 2 mantissa bits).
This is a mini-float that represents the log2 of the per-64 max. The conversion
`vcvt_bf162e6m2` rounds to the nearest e6m2 representable value, and
`vcvt_rcpe6m22bf16` computes its reciprocal as a BF16.

In the paper's terms: `Ea = round(log2(Ma))` quantized to e6m2 precision.
The offset (38 in the standard) is baked into the e6m2 bias.


## 3. Python Golden Generation

The Python golden replicates stages 1–3 (continuous case only) using NumPy:

1. Compute per-4/8/64 abs-max via reshaping.
2. Derive Ea (e6m2 quantization of log2(Ma)), Eb, Ec via the threshold checks.
3. Compute the per-4 scaling factor.
4. Quantize each BF16 element to its e1m2 4-bit code.

**It does NOT produce the Cube interleaved exponent layout** — that's a
CCE-specific data movement. The golden outputs:
- `golden_fp4.bin` — packed FP4 e1m2 codes (2 per byte)
- `golden_ea.bin` — Ea exponents (1 byte per 64 elements)
- `golden_eb.bin` — Eb bits (1 bit per 8 elements, packed)
- `golden_ec.bin` — Ec bits (1 bit per 4 elements, packed)
- `golden_scale.bin` — per-4 scaling factors (BF16)

### 3.1 Reconstruction error

A side function reconstructs the original BF16 values from the FP4 codes +
exponents and reports the max/mean relative error, for both the Python
golden and (when available) the CCE output.
