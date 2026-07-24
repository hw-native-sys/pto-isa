#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# A6 TQUANT BF16→HiFloat4 golden generator + reconstruction-error analysis.
#
# Implements the HiFloat4 quantization algorithm (arXiv:2602.11287) in pure
# NumPy, following the CCE pipeline in a6/TQuant.hpp (stages 1-3, continuous).
# Also produces the Cube-layout interleaved exponent golden (ExpLayoutForCube).
#
# GENERIC over tile size: any [M, N] where M*N % 1024 == 0 and N % 64 == 0.
# Multiple test cases are generated, each in its own subdirectory.
#
# build.py imports the DEFAULT case's manifest (CASE_NAME, ARTIFACTS, etc.)
# for config.toml generation. The golden generation covers all cases.
#
# See docs/isa/TQUANT_HIF4.md for the full algorithm derivation.
# --------------------------------------------------------------------------------

import argparse
import os
from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

np.random.seed(19)

# ============================================================
#  Artifact manifest (consumed by build.py)
# ============================================================


@dataclass(frozen=True)
class Artifact:
    """One kernel-side buffer + the file that carries its golden data.

    role:        "input" or "output"
    name:        file basename (config.toml [[*_para_array]] `name` = name + ".bin")
    para_offset: index of this buffer in the kernel launch arg list
    dtype:       platform dtype string for config.toml ("bf16", "u8", ...)
    elem_bytes:  bytes per element
    count:       number of elements
    """

    role: str
    name: str
    para_offset: int
    dtype: str
    elem_bytes: int
    count: int

    @property
    def bytes_size(self):
        return self.elem_bytes * self.count


# All kernel outputs that get verified. Order matches kernel para_offset.
OUTPUT_NAMES = ["max4", "max8", "ea", "eb", "ec", "exp_dst", "fp4", "scale"]


def make_artifacts(valid_rows, valid_cols):
    """Build the artifact manifest for a [valid_rows, valid_cols] tile.

    Constraints: (valid_rows * valid_cols) % 1024 == 0, valid_cols % 64 == 0.
    """
    total = valid_rows * valid_cols
    return [
        Artifact("input", "input1", 0, "bf16", 2, total),
        Artifact("output", "max4", 1, "bf16", 2, total // 4),
        Artifact("output", "max8", 2, "bf16", 2, total // 8),
        Artifact("output", "ea", 3, "u8", 1, (total // 64) * 2),  # 2B per Ea (zero-extended bf16)
        Artifact("output", "eb", 4, "u8", 1, (total // 8) * 2 // 8),  # psts PK: Eb 2× upsampled bits → bytes
        Artifact("output", "ec", 5, "u8", 1, (total // 4) // 8),  # psts PK: Ec packed bits → bytes
        Artifact("output", "exp_dst", 6, "u8", 1, (total // 64) * 4),
        Artifact("output", "fp4", 7, "u8", 1, total // 2),
        Artifact("output", "scale", 8, "bf16", 2, total // 2),  # half input: INTLV 2× + US_B16 2× = 4×
    ]


def make_golden_ref():
    return {
        "output_names": OUTPUT_NAMES,
        "threshold": 0.0,  # bit-exact expected
    }


# ============================================================
#  Case parameters
# ============================================================


@dataclass(frozen=True)
class Hif4Params:
    """One HiF4 test case."""

    valid_rows: int
    valid_cols: int
    kind: str = "hif4"

    def __post_init__(self):
        total = self.valid_rows * self.valid_cols
        assert total % 1024 == 0, f"M*N={total} must be divisible by 1024"
        assert self.valid_cols % 64 == 0, f"N={self.valid_cols} must be divisible by 64"

    @property
    def case_name(self):
        return f"TQUANT_HIF4_A6_TEST.case_bf16_{self.valid_rows}x{self.valid_cols}_hif4_nd"

    @property
    def artifacts(self):
        return make_artifacts(self.valid_rows, self.valid_cols)


@dataclass(frozen=True)
class VaddParams:
    """One vadd stub test case (basic connectivity probe)."""

    valid_rows: int
    valid_cols: int
    kind: str = "vadd"

    @property
    def case_name(self):
        return f"TQUANT_HIF4_A6_TEST.case_bf16_{self.valid_rows}x{self.valid_cols}_vadd_nd"

    @property
    def artifacts(self):
        total = self.valid_rows * self.valid_cols
        return [
            Artifact("input", "input1", 1, "bf16", 2, total),
            Artifact("input", "input2", 2, "bf16", 2, total),
            Artifact("output", "output", 0, "bf16", 2, total),
        ]


# The full set of test cases. build.py --case <n> selects one.
CASE_PARAMS = [
    VaddParams(128, 128),  # 0: vadd stub — basic functionality
    Hif4Params(128, 128),  # 1: HiF4 default/reference
    Hif4Params(64, 128),  # 2: HiF4 smaller
    Hif4Params(256, 128),  # 3: HiF4 medium
    Hif4Params(128, 256),  # 4: HiF4 medium wide
    Hif4Params(256, 256),  # 5: HiF4 max
    Hif4Params(128, 512),  # 6: HiF4 max wide
]

# The DEFAULT case (used when --case is not specified).
DEFAULT_PARAMS = CASE_PARAMS[1]  # 128×128 HiF4

# Module-level constants consumed by build.py (for the default case).
CASE_NAME = DEFAULT_PARAMS.case_name
VALID_ROWS = DEFAULT_PARAMS.valid_rows
VALID_COLS = DEFAULT_PARAMS.valid_cols
ARTIFACTS = DEFAULT_PARAMS.artifacts
GOLDEN_REF = (
    {"output_names": [a.name for a in ARTIFACTS if a.role == "output"], "threshold": 0.0}
    if DEFAULT_PARAMS.kind == "hif4"
    else {"output_names": ["output"], "threshold": 0.0}
)


# ============================================================
#  HiF4 constants
# ============================================================

GP4_SIZE = 4
GP8_SIZE = 8
GP64_SIZE = 64

E1M2_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float64)


# ============================================================
#  e6m2 exponent quantization
# ============================================================


def bf16_to_e6m2(ma_flat):
    """Convert per-64 abs-max to e6m2 byte codes.

    Matches CCE: vmuls(Ma, recp_7) → vcvt_bf162e6m2(ROUND_R, PART_EVEN).
    e6m2 format: 6 exp bits [7:2] + 2 mant bits [1:0], NO sign bit.
    """
    import math

    ma = np.abs(ma_flat).astype(np.float32).astype(bfloat16).astype(np.float32)
    recp_7 = np.float32(1.0 / 7.0).astype(bfloat16).astype(np.float32)
    e6m2_codes = np.zeros(len(ma), dtype=np.uint8)
    for i, v in enumerate(ma):
        if v == 0.0:
            continue
        sf = (v * recp_7).astype(bfloat16).astype(np.float32)
        if sf == 0.0:
            continue
        exp_raw = int(math.floor(math.log2(sf)))
        if exp_raw < -48:
            exp_raw = -48
        if exp_raw > 15:
            exp_raw = 15
        biased_exp = exp_raw + 48
        mantissa_frac = sf / (2.0**exp_raw)
        # Round-half-to-even (matches CCE vcvt_bf162e6m2 ROUND_R)
        mant_real = (mantissa_frac - 1.0) * 4
        mant_bits = int(round(mant_real))
        if mant_bits >= 4:
            mant_bits = 0
            biased_exp += 1
        if biased_exp > 63 or (biased_exp == 63 and mant_bits == 3):
            biased_exp = 63
            mant_bits = 2
        e6m2_codes[i] = (biased_exp << 2) | mant_bits
    return e6m2_codes


def e6m2_code_to_value(codes):
    """Convert e6m2 byte codes back to FP64 values (bias=48, normal-only)."""
    vals = np.zeros(len(codes), dtype=np.float64)
    for i, c in enumerate(codes):
        exp = int((c >> 2) & 0x3F)
        mant = int(c & 0x03)
        if exp == 0 and mant == 0:
            continue  # no zero in e6m2, but treat code 0 as min normal
        vals[i] = (1.0 + mant * 0.25) * (2.0 ** (exp - 48))
    return vals


def e6m2_code_to_reciprocal_bf16(codes):
    """Compute reciprocal of e6m2 values as BF16 (matches vcvt_rcpe6m22bf16).

    This is the TRUE FP reciprocal 1/e6m2_value (mantissa preserved via LUT),
    NOT 2^(-floor(log2(Ma))).
    """
    vals = e6m2_code_to_value(codes)
    with np.errstate(divide="ignore"):
        recips = np.where(vals > 0, 1.0 / vals, 0.0)
    return recips.astype(np.float32).astype(bfloat16)


# ============================================================
#  HiF4 quantization (continuous)
# ============================================================


def hif4_quantize(src_bf16):
    """Full HiF4 quantization of a contiguous BF16 tile.

    All arithmetic done in BF16 precision to match the CCE hardware.
    Returns dict with max4/max8 (BF16), ea/eb/ec (uint8), scale (BF16),
    fp4_codes (uint8), dequant (BF16).
    """

    # Helper: cast to BF16 after every operation (matches CCE register precision)
    def bf16(x):
        return x.astype(np.float32).astype(bfloat16).astype(np.float32)

    src = src_bf16.astype(np.float32).ravel()
    n = len(src)
    assert n % GP64_SIZE == 0, f"Total elements {n} must be divisible by 64"

    # --- Stage 1: per-4/8 abs-max (BF16 precision) ---
    abs_src = bf16(np.abs(src))
    mc = bf16(abs_src.reshape(-1, GP4_SIZE).max(axis=1))
    mb = bf16(abs_src.reshape(-1, GP8_SIZE).max(axis=1))
    ma = bf16(abs_src.reshape(-1, GP64_SIZE).max(axis=1))

    # --- Stage 2: exponent derivation ---
    # Ea: e6m2 quantization of Ma/7 (paper Algorithm 1, line 8-9)
    # CCE: vmuls(Ma, 1/7) → vcvt_bf162e6m2
    ea_codes = bf16_to_e6m2(ma)

    # Ea_rec: CCE path — vcvt_rcpe6m22bf16 (1/e6m2_value as BF16)
    ea_rec = e6m2_code_to_reciprocal_bf16(ea_codes)  # already BF16
    ea_rec_f32 = ea_rec.astype(np.float32)

    # Eb: Mb * Ea_rec >= 4 (CCE: vmul → vcmps_ge, all in BF16)
    ea_rec_per8 = np.repeat(ea_rec_f32, GP64_SIZE // GP8_SIZE)
    eb_tmp = bf16(mb * ea_rec_per8)
    eb_bits = (eb_tmp >= 4.0).astype(np.uint8)
    eb_rec = np.where(eb_bits == 1, 0.5, 1.0)

    # Ec: Mc * Ea_rec * 2^(-Eb) >= 2 (CCE: vmul → vsel → vmul → vcmps_ge)
    ea_rec_per4 = np.repeat(ea_rec_f32, GP64_SIZE // GP4_SIZE)
    eb_rec_per4 = np.repeat(eb_rec, GP8_SIZE // GP4_SIZE)
    ec_tmp_0 = bf16(mc * ea_rec_per4)
    ec_tmp_1 = bf16(ec_tmp_0 * eb_rec_per4)
    ec_bits = (ec_tmp_1 >= 2.0).astype(np.uint8)
    ec_rec = np.where(ec_bits == 1, 0.5, 1.0)

    # Scale: Ea_rec * 2^(-Eb) * 2^(-Ec) — all BF16 multiplications
    ebc_rec = bf16(eb_rec_per4 * ec_rec)
    scale = bf16(ea_rec_per4 * ebc_rec)
    scale_bf16 = scale.astype(bfloat16)

    # --- Stage 3: quantize to FP4 e1m2 ---
    scale_per_elem = np.repeat(scale, GP4_SIZE)
    scaled_src = bf16(src * scale_per_elem)
    fp4_codes = bf16_to_e1m2(scaled_src)
    dequant = dequantize_e1m2(fp4_codes, scale_per_elem).astype(bfloat16)

    return {
        "max4": mc.astype(bfloat16),
        "max8": mb.astype(bfloat16),
        "ea": ea_codes,
        "eb": eb_bits,
        "ec": ec_bits,
        "scale": scale_bf16,
        "fp4_codes": fp4_codes,
        "dequant": dequant,
    }


def bf16_to_e1m2(scaled_flat):
    """Quantize scaled BF16 values to FP4 e1m2 codes (0-15: bit3=sign, bits[2:0]=mag code).
    Matches CCE vcvt(bf16 → f4e1m2x2) with ROUND_R (round-to-nearest, ties-to-even).
    At exact midpoints between two codes, rounds to the even code."""
    vals = np.asarray(scaled_flat, dtype=np.float32)
    sign = (vals < 0).astype(np.uint8)
    mag = np.abs(vals)
    codes = np.zeros(len(vals), dtype=np.uint8)
    for i, m in enumerate(mag):
        diffs = np.abs(E1M2_VALUES - m)
        min_diff = diffs.min()
        candidates = np.where(diffs == min_diff)[0]
        if len(candidates) == 1:
            best = int(candidates[0])
        else:
            # Tie: round to even code
            c0, c1 = int(candidates[0]), int(candidates[1])
            best = c0 if c0 % 2 == 0 else c1
        codes[i] = (sign[i] << 3) | best
    return codes


def dequantize_e1m2(codes, scale_per_elem):
    """Reconstruct BF16 values from FP4 codes + per-element scale."""
    sign = (codes >> 3) & 1
    mag_code = codes & 0x07
    mag = E1M2_VALUES[mag_code]
    vals = np.where(sign == 1, -mag, mag)
    return (vals / scale_per_elem).astype(np.float32).astype(bfloat16)


def pack_fp4_codes(codes):
    """Pack FP4 nibble codes into bytes (2 per byte, little-endian nibble order)."""
    assert len(codes) % 2 == 0
    return codes[0::2].astype(np.uint8) | (codes[1::2].astype(np.uint8) << 4)


def pack_bits_lsb(bits):
    """Pack a bit array (uint8, 0 or 1) into bytes, LSB-first within each byte.
    Matches the CCE psts PK packing mode."""
    n_bytes = (len(bits) + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(len(bits)):
        if bits[i]:
            packed[i // 8] |= 1 << (i % 8)
    return packed


def pack_predicate_cce(bits, upsample=1):
    """Pack predicate bits matching CCE psts PK mode.

    The CCE loads per-8/per-4 maxima via US_B16 (upsampling 2×),
    so each bit is duplicated `upsample` times in the VL.
    psts PK then packs LSB-first into bytes.

    bits: unique per-group values (0 or 1)
    upsample: duplication factor (1=Eb with US_B16 2×, 2=Ec with US_B16 2×)
    """
    # Duplicate each bit `upsample` times (matches US_B16)
    duplicated = np.repeat(bits, upsample)
    # Pack LSB-first
    return pack_bits_lsb(duplicated)


# ============================================================
#  ExpLayoutForCube — replicate the CCE transformation
# ============================================================


def exp_layout_for_cube(ea_flat, eb_flat, ec_flat, total_elem):
    """Replicate ExpLayoutForCube_Cont: B8-interleave Ea/Eb, then block-interleave with Ec.

    The CCE loads from UB:
      - Ea via DS_B8 from zero-extended buffer (byte offset = loop_idx * 128)
        DS_B8 takes every other byte → effective packed offset = loop_idx * 64
      - Eb via DS_B8 from upsampled buffer (byte offset = loop_idx * 128)
        DS_B8 takes every other byte → effective packed offset = loop_idx * 64
      - Ec via NORM (byte offset = loop_idx * 256)
    Then vintlv(Ea, Eb) → 256B, then vsstb blockStride=2: [EaEb_blk0, Ec_blk0, ...]

    ea_flat: raw Ea bytes (1B per exponent, total/64 bytes)
    eb_flat: packed Eb bytes (after DS_B8 downsample)
    ec_flat: packed Ec bytes
    """
    input_size = total_elem // 64
    loop_num = (input_size + 127) // 128
    exp_dst = bytearray()
    for loop_idx in range(loop_num):
        ea_chunk = np.zeros(128, dtype=np.uint8)
        eb_chunk = np.zeros(128, dtype=np.uint8)
        ec_chunk = np.zeros(256, dtype=np.uint8)
        # DS_B8 offset in zero-extended space = loop_idx * 128
        # Packed offset = loop_idx * 128 / 2 = loop_idx * 64
        ea_start = loop_idx * 64
        eb_start = loop_idx * 64
        ec_start = loop_idx * 256
        ea_chunk[: min(128, len(ea_flat) - ea_start)] = ea_flat[ea_start : ea_start + 128]
        eb_chunk[: min(128, len(eb_flat) - eb_start)] = eb_flat[eb_start : eb_start + 128]
        ec_chunk[: min(256, len(ec_flat) - ec_start)] = ec_flat[ec_start : ec_start + 256]
        eaeb = np.empty(256, dtype=np.uint8)
        eaeb[0::2] = ea_chunk
        eaeb[1::2] = eb_chunk
        for blk in range(8):
            exp_dst.extend(eaeb[blk * 32 : (blk + 1) * 32].tobytes())
            exp_dst.extend(ec_chunk[blk * 32 : (blk + 1) * 32].tobytes())
    return bytes(exp_dst)


# ============================================================
#  Reconstruction error analysis
# ============================================================


def reconstruction_error(original, dequant):
    """Report max/mean relative error between original and dequantized values."""
    orig = original.ravel().astype(np.float64)
    deq = dequant.ravel().astype(np.float64)
    abs_err = np.abs(orig - deq)
    denom = np.maximum(np.abs(orig), np.abs(deq))
    denom = np.where(denom > 0, denom, 1.0)
    rel_err = abs_err / denom
    nonzero_mask = np.abs(orig) > 1e-10
    rel_nz = rel_err[nonzero_mask] if nonzero_mask.any() else np.array([0.0])
    print(f"  Abs error:  max={abs_err.max():.4e}  mean={abs_err.mean():.4e}")
    print(f"  Rel error:  max={rel_err.max():.4e}  mean={rel_err.mean():.4e}")
    print(f"  Rel (nz):   max={rel_nz.max():.4e}  mean={rel_nz.mean():.4e}")
    print(f"  RMSE: {np.sqrt((abs_err**2).mean()):.4e}")
    print(f"  SNR: {20 * np.log10(np.abs(orig).max() / np.sqrt((abs_err**2).mean())):.2f} dB")


# ============================================================
#  CCE comparison — load CCE outputs and reconstruct
# ============================================================


def load_cce_outputs(case_params, cce_dir="."):
    """Load CCE-produced .bin outputs and reconstruct the original BF16 values.

    The CCE stores:
      ea.bin:       Ea as zero-extended BF16 (2B per exponent) → unpack to e6m2 codes
      eb.bin:       Eb as packed bits, upsampled 2× → unpack to per-8-group bits
      ec.bin:       Ec as packed bits → unpack to per-4-group bits
      fp4.bin:      packed FP4 e1m2 codes (2 per byte)
      scale.bin:    per-element BF16 scaling (half input size, INTLV 2× + US_B16 2×)

    Reconstruct: value = e1m2_to_value(fp4_code) / scale
    """
    total = case_params.valid_rows * case_params.valid_cols

    # Load FP4 codes
    fp4_packed = np.fromfile(os.path.join(cce_dir, "fp4.bin"), dtype=np.uint8)
    codes_lo = fp4_packed & 0x0F
    codes_hi = (fp4_packed >> 4) & 0x0F
    fp4_codes = np.empty(len(fp4_packed) * 2, dtype=np.uint8)
    fp4_codes[0::2] = codes_lo
    fp4_codes[1::2] = codes_hi
    fp4_codes = fp4_codes[:total]

    # Load scaling (half input size: each per-2-element BF16, upsampled 2× at load)
    scale_raw = np.fromfile(os.path.join(cce_dir, "scale.bin"), dtype=bfloat16)
    # The stored scale has each per-4 value repeated 2× (INTLV). Upsample 2× more to get per-element.
    scale_per_elem = np.repeat(scale_raw, 2)[:total]

    # Reconstruct: sign + magnitude from fp4 code, divided by scale
    sign = (fp4_codes >> 3) & 1
    mag_code = fp4_codes & 0x07
    mag = E1M2_VALUES[mag_code]
    vals = np.where(sign == 1, -mag, mag)
    # scale_per_elem is the inverse scale stored by CCE (Ea_rec * 2^(-Eb) * 2^(-Ec))
    # The forward scale applied was: input * scale → fp4. So input = fp4_value / scale.
    dequant = vals / scale_per_elem.astype(np.float64)

    return dequant


def compare_cce_vs_golden(case_params, input_bin, cce_dir="."):
    """Compare reconstruction quality: CCE outputs vs Python golden, both against the
    original BF16 input."""
    total = case_params.valid_rows * case_params.valid_cols

    # Load original input
    src = np.fromfile(input_bin, dtype=bfloat16).astype(np.float64)

    # Python golden reconstruction
    src_2d = src.reshape(case_params.valid_rows, case_params.valid_cols)
    golden_result = hif4_quantize(src_2d.astype(bfloat16))
    golden_dequant = golden_result["dequant"]

    # CCE reconstruction
    cce_dequant = load_cce_outputs(case_params, cce_dir)

    print("=" * 70)
    print("CCE vs Golden Reconstruction Comparison")
    print(f"  Input: {case_params.valid_rows}×{case_params.valid_cols} = {total} elements")
    print("=" * 70)

    print("\n--- Python Golden ---")
    reconstruction_error(src, golden_dequant)

    print("\n--- CCE Output ---")
    reconstruction_error(src, cce_dequant)

    # Direct comparison between golden and CCE
    diff = np.abs(golden_dequant - cce_dequant)
    print("\n--- Golden vs CCE (direct) ---")
    print(f"  Max diff:  {diff.max():.6e}")
    print(f"  Mean diff: {diff.mean():.6e}")
    print(f"  Matching elements: {(diff < 1e-6).sum()}/{total} ({100 * (diff < 1e-6).mean():.1f}%)")


# ============================================================
#  Input generation
# ============================================================


def make_bf16_input(valid_rows, valid_cols):
    """Generate BF16 input with wide dynamic range and per-group variation.

    Each 64-element group gets a random scale in [2^-6, 2^6] so adjacent
    groups have different max magnitudes — exercising the per-64/8/4 exponent
    hierarchy. Values stay well within BF16 range (max ~10K).
    """
    total = valid_rows * valid_cols
    rng = np.random.default_rng(19)

    # Random mantissa values in [-1, 1] — small base magnitudes
    base = rng.uniform(-1.0, 1.0, size=total).astype(np.float32)

    # Per-64-group power-of-2 scale: random exponents in [-7, 17]
    # so values span roughly [-100000, 100000] (2^17 ≈ 131072)
    gp64 = 64
    num_groups = (total + gp64 - 1) // gp64
    scales = np.ones(total, dtype=np.float32)
    for g in range(num_groups):
        begin = g * gp64
        end = min(begin + gp64, total)
        exp = rng.integers(-7, 18)  # 2^-7 .. 2^17
        scales[begin:end] = np.ldexp(np.float32(1.0), int(exp))

    values = base * scales
    return values.reshape(valid_rows, valid_cols).astype(bfloat16)


# ============================================================
#  Per-case golden generation
# ============================================================


def gen_golden_for_case(params):
    """Generate all artifacts for one case (HiF4 or vadd) into the current directory."""
    if params.kind == "vadd":
        return gen_golden_vadd(params)
    return gen_golden_hif4(params)


def gen_golden_vadd(params):
    """Generate vadd stub artifacts: two BF16 inputs + golden (src0+src1)."""
    src0 = make_bf16_input(params.valid_rows, params.valid_cols)
    src1 = make_bf16_input(params.valid_rows, params.valid_cols)
    golden = (src0.astype(np.float32) + src1.astype(np.float32)).astype(bfloat16)

    src0.ravel().tofile("input1.bin")
    src1.ravel().tofile("input2.bin")
    golden.ravel().tofile("golden.bin")

    for a in params.artifacts:
        data = {"input1": src0, "input2": src1, "output": golden}[a.name]
        actual = data.ravel().nbytes
        ok = actual == a.bytes_size
        print(f"  {a.name + '.bin':20s} {actual:6d} B  {'OK' if ok else 'MISMATCH'}")

    print(f"\n  vadd stub: {params.valid_rows}×{params.valid_cols} BF16")
    return True


def gen_golden_hif4(params):
    """Generate all HiF4 artifacts for one case into the current directory."""
    valid_rows = params.valid_rows
    valid_cols = params.valid_cols
    total = valid_rows * valid_cols

    src = make_bf16_input(valid_rows, valid_cols)
    result = hif4_quantize(src)

    # Pack Eb/Ec bits matching CCE psts PK (LSB-first, with US_B16 duplication)
    eb_packed = pack_predicate_cce(result["eb"], upsample=2)
    ec_packed = pack_predicate_cce(result["ec"], upsample=1)

    # For exp_dst, the CCE applies DS_B8 on Eb (downsamples the 2× upsampled data)
    eb_packed_ds = eb_packed[0::2]  # DS_B8: take every other byte

    # Cube-layout exponents: B8-interleave Ea + Eb(DS_B8'd), then block-interleave with Ec
    exp_dst = exp_layout_for_cube(result["ea"], eb_packed_ds, ec_packed, total)

    # Write all artifacts
    src.ravel().tofile("input1.bin")
    result["max4"].ravel().tofile("golden_max4.bin")
    result["max8"].ravel().tofile("golden_max8.bin")
    # Ea stored as zero-extended BF16 (Ea, 0, Ea, 0, ...) — 2B per exponent
    ea_zext = np.zeros(len(result["ea"]) * 2, dtype=np.uint8)
    ea_zext[0::2] = result["ea"]
    ea_zext.tofile("golden_ea.bin")
    # Eb stored as packed predicate bits (psts PK: LSB-first, US_B16 2× duplication)
    eb_packed.tofile("golden_eb.bin")
    ec_packed.tofile("golden_ec.bin")
    np.frombuffer(exp_dst, dtype=np.uint8).tofile("golden_exp_dst.bin")
    pack_fp4_codes(result["fp4_codes"]).tofile("golden_fp4.bin")
    # Scaling upsampled 2× at store (INTLV_B16), another 2× at load (US_B16).
    # The stored buffer has each per-4 scale repeated 2×.
    scale_upsampled = np.repeat(result["scale"], 2)
    scale_upsampled.ravel().tofile("golden_scale.bin")

    # Verify sizes
    artifacts = params.artifacts
    files = {
        "input1": src.ravel(),
        "max4": result["max4"].ravel(),
        "max8": result["max8"].ravel(),
        "ea": ea_zext,
        "eb": eb_packed,
        "ec": ec_packed,
        "exp_dst": np.frombuffer(exp_dst, dtype=np.uint8),
        "fp4": pack_fp4_codes(result["fp4_codes"]),
        "scale": scale_upsampled.ravel(),
    }
    all_ok = True
    for a in artifacts:
        data = files[a.name]
        actual = data.nbytes if hasattr(data, "nbytes") else len(data)
        ok = actual == a.bytes_size
        if not ok:
            all_ok = False
        print(f"  {a.name + '.bin':20s} {actual:6d} B  {'OK' if ok else f'MISMATCH (expected {a.bytes_size})'}")

    # Reconstruction error
    print("\n  Reconstruction error:")
    reconstruction_error(src, result["dequant"])
    print(
        f"  Eb: {result['eb'].sum()}/{len(result['eb'])} ({100 * result['eb'].mean():.1f}%)  "
        f"Ec: {result['ec'].sum()}/{len(result['ec'])} ({100 * result['ec'].mean():.1f}%)"
    )

    return all_ok


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A6 TQUANT HiF4 golden generator")
    parser.add_argument(
        "--regression", action="store_true", help="regression-platform mode: generate only the default case into cwd"
    )
    parser.add_argument("--all", action="store_true", help="generate ALL cases, each into its own subdirectory")
    parser.add_argument("--case", type=int, default=None, help="generate a specific case by index (0=vadd, 1+=hif4)")
    parser.add_argument(
        "--compare-cce", type=str, default=None, help="compare CCE outputs in the given directory against golden"
    )
    parser.add_argument(
        "--cce-case", type=int, default=1, help="case index for CCE comparison (default: 1 = hif4 128x128)"
    )
    args = parser.parse_args()

    if args.compare_cce:
        params = CASE_PARAMS[args.cce_case]
        input_bin = os.path.join(args.compare_cce, "input1.bin")
        if not os.path.exists(input_bin):
            # Fall back to the golden input
            input_bin = "input1.bin"
        compare_cce_vs_golden(params, input_bin, args.compare_cce)
    elif args.case is not None:
        params = CASE_PARAMS[args.case]
        print(f"  Case [{args.case}]: {params.case_name}")
        print(f"  Kind: {params.kind}  Size: {params.valid_rows}×{params.valid_cols}")
        gen_golden_for_case(params)
    elif args.all:
        print("=" * 70)
        print(f"Generating goldens for {len(CASE_PARAMS)} cases")
        print("=" * 70)
        for i, params in enumerate(CASE_PARAMS):
            print(f"\n{'=' * 70}")
            print(f"  Case [{i}]: {params.case_name}")
            print(f"  Kind: {params.kind}  Size: {params.valid_rows}×{params.valid_cols}")
            print(f"{'=' * 70}")
            case_dir = params.case_name.split(".")[-1]
            if not os.path.exists(case_dir):
                os.makedirs(case_dir)
            original_dir = os.getcwd()
            os.chdir(case_dir)
            gen_golden_for_case(params)
            os.chdir(original_dir)
        print(f"\nAll {len(CASE_PARAMS)} cases generated.")
    elif args.regression:
        print("=" * 70)
        print(f"Golden Generation (default case: {DEFAULT_PARAMS.case_name})")
        print(f"  Kind: {DEFAULT_PARAMS.kind}  Size: {VALID_ROWS}×{VALID_COLS}")
        print("=" * 70)
        gen_golden_for_case(DEFAULT_PARAMS)
        print("\nGolden generation complete.")
    else:
        # Default (no args): generate ALL cases into <Suite.Case>/ subdirectories.
        # This matches the run_st.py flow: gen_data runs from build/, creates
        # build/<casedir>/ for each case, binary runs from build/bin/ reading ../<casedir>/
        print("=" * 70)
        print(f"Generating goldens for {len(CASE_PARAMS)} cases")
        print("=" * 70)
        for i, params in enumerate(CASE_PARAMS):
            print(f"\n{'=' * 70}")
            print(f"  Case [{i}]: {params.case_name}")
            print(f"  Kind: {params.kind}  Size: {params.valid_rows}x{params.valid_cols}")
            print(f"{'=' * 70}")
            case_dir = params.case_name  # e.g. TQUANT_HIF4_A6_TEST.case_bf16_128x128_hif4_nd
            if not os.path.exists(case_dir):
                os.makedirs(case_dir)
            original_dir = os.getcwd()
            os.chdir(case_dir)
            gen_golden_for_case(params)
            os.chdir(original_dir)
        print(f"\nAll {len(CASE_PARAMS)} cases generated.")
