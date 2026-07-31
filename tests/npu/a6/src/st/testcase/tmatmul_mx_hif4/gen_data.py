#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# A6 HiF4 Cube matmul golden generator.
# --------------------------------------------------------------------------------

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

np.random.seed(19)

GP4_SIZE = 4
GP8_SIZE = 8
GP64_SIZE = 64
HIF4_SCALE_GROUP = 64
E1M2_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float64)


# ============================================================
#  Artifact manifest (consumed by build.py)
# ============================================================


@dataclass(frozen=True)
class Artifact:
    role: str
    name: str
    para_offset: int
    dtype: str
    elem_bytes: int
    count: int

    @property
    def bytes_size(self):
        return self.elem_bytes * self.count


@dataclass(frozen=True)
class Hif4MatmulParams:
    valid_m: int
    valid_k: int
    valid_n: int

    def __post_init__(self):
        assert self.valid_m % 16 == 0, f"M={self.valid_m} must be divisible by 16"
        assert self.valid_k % 64 == 0, f"K={self.valid_k} must be divisible by 64"
        assert self.valid_n % 64 == 0, f"N={self.valid_n} must be divisible by 64"

    @property
    def case_name(self):
        return f"TMATMUL_MX_HIF4_A6_TEST.case_hif4_{self.valid_m}x{self.valid_k}x{self.valid_n}_nd"

    @property
    def artifacts(self):
        m, k, n = self.valid_m, self.valid_k, self.valid_n
        total_a = m * k
        total_b = k * n
        total_out = m * n
        return [
            Artifact("output", "out", 0, "bf16", 2, total_out),
            Artifact("input", "a_data", 1, "u8", 1, total_a // 2),
            Artifact("input", "a_scale", 2, "u8", 1, (total_a // 64) * 4),
            Artifact("input", "b_data", 3, "u8", 1, total_b // 2),
            Artifact("input", "b_scale", 4, "u8", 1, (total_b // 64) * 4),
        ]


# ============================================================
#  HiF4 quantization primitives (from tquant/gen_data.py)
# ============================================================


def _bf16(x):
    return x.astype(np.float32).astype(bfloat16).astype(np.float32)


def bf16_to_e6m2(ma_flat):
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
    vals = np.zeros(len(codes), dtype=np.float64)
    for i, c in enumerate(codes):
        exp = int((c >> 2) & 0x3F)
        mant = int(c & 0x03)
        if exp == 0 and mant == 0:
            continue
        vals[i] = (1.0 + mant * 0.25) * (2.0 ** (exp - 48))
    return vals


def e6m2_code_to_reciprocal_bf16(codes):
    vals = e6m2_code_to_value(codes)
    with np.errstate(divide="ignore"):
        recips = np.where(vals > 0, 1.0 / vals, 0.0)
    return recips.astype(np.float32).astype(bfloat16)


def bf16_to_e1m2(scaled_flat):
    """Quantize scaled BF16 values to FP4 e1m2 codes (0-15: bit3=sign,
    bits[2:0]=mag code). Matches CCE vcvt(bf16 → f4e1m2x2) with ROUND_R
    (round-to-nearest, ties-to-even). At exact midpoints between two codes,
    rounds to the even code.

    NOTE: the 2^Ng pre-scale variant (rounding in [0,7]) diverged from the
    CCE TQuant hardware by ~1.3% of nibbles on pure tie-break cases and did
    not change the matmul RMSE — so it has been reverted in favour of the
    CCE-exact quantization used by tquant/gen_data.py.
    """
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


def hif4_quantize(src_bf16):
    src = src_bf16.astype(np.float32).ravel()
    n = len(src)
    assert n % GP64_SIZE == 0, f"Total elements {n} must be divisible by 64"

    abs_src = _bf16(np.abs(src))
    mc = _bf16(abs_src.reshape(-1, GP4_SIZE).max(axis=1))
    mb = _bf16(abs_src.reshape(-1, GP8_SIZE).max(axis=1))
    ma = _bf16(abs_src.reshape(-1, GP64_SIZE).max(axis=1))

    ea_codes = bf16_to_e6m2(ma)
    ea_rec = e6m2_code_to_reciprocal_bf16(ea_codes)
    ea_rec_f32 = ea_rec.astype(np.float32)

    ea_rec_per8 = np.repeat(ea_rec_f32, GP64_SIZE // GP8_SIZE)
    eb_tmp = _bf16(mb * ea_rec_per8)
    eb_bits = (eb_tmp >= 4.0).astype(np.uint8)
    eb_rec = np.where(eb_bits == 1, 0.5, 1.0)

    ea_rec_per4 = np.repeat(ea_rec_f32, GP64_SIZE // GP4_SIZE)
    eb_rec_per4 = np.repeat(eb_rec, GP8_SIZE // GP4_SIZE)
    ec_tmp_0 = _bf16(mc * ea_rec_per4)
    ec_tmp_1 = _bf16(ec_tmp_0 * eb_rec_per4)
    ec_bits = (ec_tmp_1 >= 2.0).astype(np.uint8)
    ec_rec = np.where(ec_bits == 1, 0.5, 1.0)

    ebc_rec = _bf16(eb_rec_per4 * ec_rec)
    scale = _bf16(ea_rec_per4 * ebc_rec)

    # CCE-exact path: quantize in the [0, 1.75] range (no 2^Ng pre-scale).
    # The CCE TQuant hardware applies vcvt(bf16 -> f4e1m2x2) with ROUND_R on
    # the unscaled, BF16-multiplied value. See bf16_to_e1m2 docstring for why
    # the 2^Ng variant was reverted.
    scale_per_elem = np.repeat(scale, GP4_SIZE)
    scaled_src = _bf16(src * scale_per_elem)
    fp4_codes = bf16_to_e1m2(scaled_src)
    dequant = dequantize_e1m2(fp4_codes, scale_per_elem).astype(bfloat16)

    return {"ea": ea_codes, "eb": eb_bits, "ec": ec_bits, "fp4_codes": fp4_codes, "dequant": dequant}


def exp_layout_for_cube(ea_flat, eb_flat, ec_flat, total_elem):
    """Build the Cube scale layout: B8-interleave Ea/Eb, then block-interleave with Ec.

    This is the golden (paper-faithful) version. The CCE's ExpLayoutForCube_Cont
    applies DS_B8 to Eb (dropping Eb bits b4-b7 of every group); that quirk does
    NOT match the paper, so eb_flat here is expected to carry all 8 Eb bits per
    group (one byte per group via pack_bits_lsb). Layout: vintlv(Ea, Eb) -> 256B,
    then vsstb blockStride=2: [EaEb_blk0, Ec_blk0, ...].

    ea_flat: raw Ea bytes (1B per exponent, total/64 bytes)
    eb_flat: packed Eb bytes (1B per group, all 8 Eb bits, paper-faithful)
    ec_flat: packed Ec bytes
    """
    input_size = total_elem // 64
    loop_num = (input_size + 127) // 128
    exp_dst = bytearray()
    for loop_idx in range(loop_num):
        ea_chunk = np.zeros(128, dtype=np.uint8)
        eb_chunk = np.zeros(128, dtype=np.uint8)
        ec_chunk = np.zeros(256, dtype=np.uint8)
        ea_start = loop_idx * 64  # CCE DS_B8 halves the loop_idx*128 offset
        eb_start = loop_idx * 128  # CCE NORM: no halving (was *64 when Eb used DS_B8)
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


def pack_bits_lsb(bits):
    """Pack a bit array (uint8, 0 or 1) into bytes, LSB-first within each byte.
    Used for both Eb (8 bits per 64-group -> 1 byte) and Ec (16 bits per
    64-group -> 2 bytes). Paper-faithful: keeps all bits, no CCE US_B16/DS_B8."""
    n_bytes = (len(bits) + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(len(bits)):
        if bits[i]:
            packed[i // 8] |= 1 << (i % 8)
    return packed


# ============================================================
#  Matmul golden generation
# ============================================================


# Module-level RNG so A and B get different values. Seeded for reproducibility:
# the first make_bf16_matrix() call consumes the first chunk, the second call
# consumes the next chunk — never the same values.
_BF16_RNG = np.random.default_rng(19)


def make_bf16_matrix(valid_m, valid_n, group_axis="row"):
    """Generate BF16 input with values in [-10, 10] and small per-64-group variation.

    Realistic for Q/K/V in Flash Attention (post-LayerNorm + Linear, with room
    for outliers). Each 64-element group gets a random scale factor so adjacent
    groups have slightly different magnitudes — enough to exercise the HiF4
    Ea/Eb/Ec hierarchy without the extreme dynamic range (2^21) that caused
    edge-case rounding divergence between the CCE and the numpy reference.

    group_axis controls which axis the 64-element groups run along:
      - "row" (default): groups are 64 contiguous elements along the row
        (N-axis for A[M,K]; matches ravel() of a row-major matrix)
      - "col": groups are 64 contiguous elements along the column (K-axis).
        Used for B so the scale groups align with the matmul contraction
        direction (the matmul reads B along K for each N-column).

    Uses a module-level RNG so consecutive calls (A then B) get DIFFERENT
    matrices — a per-call seed would silently make A == B.
    """
    total = valid_m * valid_n
    base = _BF16_RNG.uniform(-1.0, 1.0, size=total).astype(np.float32)

    if group_axis == "row":
        # Groups walk the ravel() (row-major: cols fastest). Same as A.
        scales = np.ones(total, dtype=np.float32)
        gp64 = 64
        num_groups = (total + gp64 - 1) // gp64
        for g in range(num_groups):
            begin = g * gp64
            end = min(begin + gp64, total)
            # Per-group scale in [0.5, 10] — realistic attention range
            scales[begin:end] = _BF16_RNG.uniform(0.5, 10.0)
        values = base * scales
        return values.reshape(valid_m, valid_n).astype(bfloat16)

    # group_axis == "col": groups run down columns. Generate in transposed
    # orientation (so ravel walks the contraction axis), apply scales,
    # then transpose back to [valid_m, valid_n].
    values_t = base.reshape(valid_n, valid_m)  # [N, M] view (cols are now rows)
    scales = np.ones(total, dtype=np.float32)
    gp64 = 64
    num_groups = (total + gp64 - 1) // gp64
    for g in range(num_groups):
        begin = g * gp64
        end = min(begin + gp64, total)
        scales[begin:end] = _BF16_RNG.uniform(0.5, 10.0)
    values_t = values_t * scales.reshape(valid_n, valid_m)
    # Transpose back to [valid_m, valid_n] = [K, N] for B
    return values_t.T.astype(bfloat16)


def pack_fp4_nd(fp4_codes_1d):
    flat = np.asarray(fp4_codes_1d, dtype=np.uint8)
    assert len(flat) % 2 == 0
    fp4_bytes = bytearray()
    for i in range(0, len(flat), 2):
        lo = int(flat[i]) & 0x0F
        hi = int(flat[i + 1]) & 0x0F
        fp4_bytes.append(lo | (hi << 4))
    return bytes(fp4_bytes)


def quantize_to_hif4_artifacts(bf16_mat):
    """Quantize BF16 matrix to HiF4 artifacts: FP4 data + scale bytes.

    Follows the paper (Algorithm 1), NOT the CCE's DS_B8 quirk:
      - Eb: all 8 per-8-subgroup bits per 64-group packed into ONE byte
        (LSB-first via pack_bits_lsb). The CCE upsamples Eb 2x via US_B16 then
        drops the upper half with DS_B8, silently discarding Eb bits b4-b7;
        that does NOT match the paper, so we keep all 8 bits per group.
      - Ec: all 16 per-4-subgroup bits per 64-group packed into 2 bytes
        (LSB-first via pack_bits_lsb).
      - exp_layout_for_cube interleaves Ea/Eb/Ec.
    """
    res = hif4_quantize(bf16_mat)
    total_elem = bf16_mat.size

    # Eb: one byte per 64-group carrying all 8 Eb bits (paper-faithful, no DS_B8)
    eb_packed = pack_bits_lsb(res["eb"])
    ec_packed = pack_bits_lsb(res["ec"])

    fp4_data = pack_fp4_nd(res["fp4_codes"])
    scale_bytes = exp_layout_for_cube(res["ea"], eb_packed, ec_packed, total_elem)
    return fp4_data, scale_bytes


def _build_hif4_scale_patch_layout(ea, eb, ec, rows, cols):
    """Build the L0AMX/L0BMX byte layout for HiF4 scale.

    Each 64B patch corresponds to one [16, 64]-element block of L0A/L0B (i.e.
    one M- or N-fractal × one K-group). The 16 rows in the patch are the 16
    M- or N-rows of that fractal; the K-group is fixed per patch.

    Patch internal layout (verified against the user's spec):
      bytes  0..31: [Ea(g0), Eb(g0), Ea(g1), Eb(g1), ... × 16 groups]  (EaEb half)
      bytes 32..63: [Ec_lo(g0), Ec_hi(g0), ... × 16 groups]            (Ec half)

    The 4 bytes per group are NOT contiguous — Ea/Eb come first as a 32B
    block, then Ec as a separate 32B block. Each Eb byte carries all 8 Eb
    bits for that 64-group (paper-faithful, no DS_B8 downsampling).

    Args:
      ea, eb, ec: per-group scale arrays, indexed by linear group id
        g_lin = walk over the source matrix. For A[M, K] row-major,
        g_lin = m * (K//64) + k_group. For B[K, N] grouped along K
        (after transpose), g_lin = n * (K//64) + k_group.
        eb is 1 byte per group (all 8 Eb bits); ec is 2 bytes per group.
      rows, cols: matrix shape. For A: (M, K). For B: (N, K) — pass the
        shape such that the "row" axis is the M- or N-axis (16-row fractal)
        and "cols" is K.

    Returns:
      bytes of length (rows//16) * (cols//64) * 64.
    """
    row_fractals = rows // 16
    k_groups = cols // 64
    out = np.zeros(row_fractals * k_groups * 64, dtype=np.uint8)
    # [row_fractals, k_groups, 2, 16, 2]: dim2=0 EaEb / 1 Ec; dim3=row; dim4=byte
    view = out.reshape(row_fractals, k_groups, 2, 16, 2)
    for rf in range(row_fractals):
        for kg in range(k_groups):
            for r in range(16):
                g_lin = (rf * 16 + r) * k_groups + kg
                view[rf, kg, 0, r, 0] = ea[g_lin]  # Ea
                view[rf, kg, 0, r, 1] = eb[g_lin]  # Eb
                view[rf, kg, 1, r, 0] = ec[g_lin * 2]  # Ec_lo
                view[rf, kg, 1, r, 1] = ec[g_lin * 2 + 1]  # Ec_hi
    return out.tobytes()


def build_hif4_scale_a_zz(a_bf16):
    """Build the HIF4_A_ZZ scale bytes for A[rows, cols].

    L0AMX shape: [rows/16, cols/64, 2, 16, 2]. Each patch (mf, kg) covers 16
    rows × one cols-group. Groups are walked as g_lin = m * (cols//64) + k_group.
    """
    valid_m, valid_k = a_bf16.shape
    res = hif4_quantize(a_bf16)
    eb_packed = pack_bits_lsb(res["eb"])  # all 8 Eb bits per group (paper-faithful)
    ec_packed = pack_bits_lsb(res["ec"])
    return _build_hif4_scale_patch_layout(res["ea"], eb_packed, ec_packed, valid_m, valid_k)


def build_hif4_scale_b_nn(b_bf16):
    """Build the HIF4_B_NN scale bytes for B[rows, cols] (logical [K, N]).

    L0BMX shape: [cols/16, rows/64, 2, 16, 2]. Each patch (nf, kg) covers 16
    cols-columns × one rows-group. The matmul reads B along K for each N, so the
    scale must group 64 consecutive K-elements per N-column.

    B must already be generated with make_bf16_matrix(K, N, group_axis="col")
    so the per-64 scale groups run along K (the contraction axis). Quantizing
    B.T (shape [N, K]) makes ravel walk K fastest, so each 64-element group is
    64 K-values for one N-column. Only the SCALE path uses B.T; the FP4 DATA
    path quantizes B directly to preserve the [K, N] GM layout.
    """
    valid_k, valid_n = b_bf16.shape
    b_t = b_bf16.T.copy()  # [N, K] — quantize so groups walk K-direction per N
    res = hif4_quantize(b_t)
    eb_packed = pack_bits_lsb(res["eb"])  # all 8 Eb bits per group (paper-faithful)
    ec_packed = pack_bits_lsb(res["ec"])
    # Pass (N, K): rows=N (fractal axis), cols=K (k_group axis)
    return _build_hif4_scale_patch_layout(res["ea"], eb_packed, ec_packed, valid_n, valid_k)


def quantize_to_hif4_a(a_bf16):
    """FP4 data (row-major ND) + HIF4_A_ZZ scale bytes for A[M, K]."""
    res = hif4_quantize(a_bf16)
    fp4_data = pack_fp4_nd(res["fp4_codes"])
    scale_bytes = build_hif4_scale_a_zz(a_bf16)
    return fp4_data, scale_bytes


def quantize_to_hif4_b(b_bf16):
    """FP4 data (row-major ND, B as [K, N]) + HIF4_B_NN scale bytes for B[K, N].

    B must be generated with `make_bf16_matrix(K, N, group_axis="col")` so
    the per-64 scale groups run along K. Both the FP4 data path AND the scale
    path must group along K (the matmul contraction axis) so the nibbles and
    scale bytes are consistent. This requires quantizing B.T (shape [N, K],
    whose ravel walks K) for the FP4 nibbles too, then transposing the nibbles
    back to [K, N] for GM storage.
    """
    b_t = b_bf16.T.copy()  # [N, K] — ravel walks K-direction
    res = hif4_quantize(b_t)
    fp4_codes = res["fp4_codes"].reshape(b_t.shape).T.copy()  # back to [K, N]
    fp4_data = pack_fp4_nd(fp4_codes.ravel())
    scale_bytes = build_hif4_scale_b_nn(b_bf16)
    return fp4_data, scale_bytes


def dequantize_for_matmul(bf16_mat):
    """Paper-faithful dequant for the matmul golden (Equation 2):

        V_i = E6M2 × 2^(E1_8[ceil(i/8)] + E1_16[ceil(i/4)]) × S1P2_i

    All 8 Eb bits per group are used (no DS_B8 downsampling). Each level is
    applied as a separate BF16-truncated multiply to mirror the mad_mx
    micro-op sequence: fp4×Ea → BF16 → ×(2^Eb) → BF16 → ×(2^Ec) → BF16.
    """
    res = hif4_quantize(bf16_mat)
    fp4_codes = res["fp4_codes"]
    n = bf16_mat.size

    # Ea value (per 64-group), broadcast to every element
    ea_vals = e6m2_code_to_value(res["ea"]).astype(np.float32)
    ea_per_elem = np.repeat(ea_vals, GP64_SIZE)[:n]

    # Eb: 2^(E1_8) per 8-element subgroup (all 8 bits, no DS_B8)
    eb_per_elem = np.repeat(res["eb"], GP8_SIZE)[:n]
    eb_factor = np.where(eb_per_elem == 1, 2.0, 1.0).astype(np.float32)

    # Ec: 2^(E1_16) per 4-element subgroup
    ec_per_elem = np.repeat(res["ec"], GP4_SIZE)[:n]
    ec_factor = np.where(ec_per_elem == 1, 2.0, 1.0).astype(np.float32)

    # FP4 magnitude with sign
    sign = (fp4_codes >> 3) & 1
    mag_code = fp4_codes & 0x07
    mag = E1M2_VALUES[mag_code].astype(np.float32)
    fp4_vals = np.where(sign == 1, -mag, mag)

    # Apply the three-level scale with BF16 truncation between each step
    dequant = (fp4_vals * ea_per_elem).astype(bfloat16).astype(np.float32)
    dequant = (dequant * eb_factor).astype(bfloat16).astype(np.float32)
    dequant = (dequant * ec_factor).astype(bfloat16).astype(np.float32)
    return dequant.reshape(bf16_mat.shape).astype(bfloat16)


def reference_matmul(a_bf16, b_bf16):
    # A groups along K (A is row-major, ravel walks K-direction). B is
    # generated with group_axis="col" so its per-64 groups also run along K.
    # Quantize B.T (shape [N, K]) so ravel walks K-direction — matches the
    # scale grouping used in build_hif4_scale_b_nn.
    a_deq = dequantize_for_matmul(a_bf16).astype(np.float32)
    b_deq = dequantize_for_matmul(b_bf16.T.copy()).T.astype(np.float32)
    c = (a_deq @ b_deq).astype(np.float32)
    return c.astype(bfloat16)


def reference_matmul_fp32(a_bf16, b_bf16):
    """FP32 accumulator before the fixpipe cast. Matches L0C contents (post-MMAD_MX)."""
    a_deq = dequantize_for_matmul(a_bf16).astype(np.float32)
    b_deq = dequantize_for_matmul(b_bf16.T.copy()).T.astype(np.float32)
    return (a_deq @ b_deq).astype(np.float32)


def _unpack_nibbles(nd_bytes, rows, cols):
    raw = np.frombuffer(nd_bytes, dtype=np.uint8)
    assert raw.size * 2 == rows * cols, f"size mismatch: {raw.size}B vs {rows}x{cols} elems"
    raw = raw.reshape(rows, cols // 2)
    elems = np.empty((rows, cols), dtype=np.uint8)
    elems[:, 0::2] = raw & 0x0F
    elems[:, 1::2] = (raw >> 4) & 0x0F
    return elems


def _pack_nibbles(elems):
    assert elems.shape[-1] % 2 == 0
    lo = elems[..., 0::2] & 0x0F
    hi = elems[..., 1::2] & 0x0F
    return (lo | (hi << 4)).astype(np.uint8)


def nd_to_nz(nd_bytes, rows, cols):
    """ND [rows, cols] -> NZ layout for FP4/HiF4 (b8 storage, c0=64 elements = 32B).

    Authoritative formula (matches A5 tquant/gen_data.py:95 `nd2nz_mxfp8`,
    adapted for FP4 where 2 nibbles pack per byte so c0_elem=64):

        elems.reshape(rows, cols//64, 64).transpose([1, 0, 2])

    Result is [cols/64, rows, 64] elements = [cols/64, rows, 32] bytes.
    NO 16-row sub-blocking on the M axis — the 16-row fractal only matters for
    tile addressing, not for the ND2NZ byte permutation. Verified byte-for-byte
    against A5 tmatmul_mx case20 (fp4_e1m2x2 128x128x128) l1.wr_log.dump.

    rows = M for A tile, K for B tile. cols = K for A tile, N for B tile.
    """
    assert cols % 64 == 0, f"need cols%64==0, got {cols}"
    elems = _unpack_nibbles(nd_bytes, rows, cols)
    nz = elems.reshape(rows, cols // 64, 64).transpose([1, 0, 2])  # [cols/64, rows, 64]
    return _pack_nibbles(nz).tobytes()


def nd_to_zn(nd_bytes, rows, cols):
    """L0B ZN layout for an FP4/HiF4 tile of logical shape [rows, cols] (e.g. [K, N]).

    ZN is the transpose of NZ: same fractal permutation applied to the transposed
    matrix. For the B tile (logical [K, N]) this means treating N as the new "rows"
    and K as the new "cols", then applying nd_to_nz.
    Verified against A5 case4 (fp4_e2m1x2, K=N=64) l0b.wr_log.dump.
    """
    elems = _unpack_nibbles(nd_bytes, rows, cols)  # [rows, cols] = [K, N]
    elems_t = elems.transpose()  # [N, K]
    nd_t_bytes = _pack_nibbles(elems_t).tobytes()
    return nd_to_nz(nd_t_bytes, rows=cols, cols=rows)


def fp32_nd_to_l0c(c_fp32_bytes, valid_m, valid_n):
    """L0C FP32 fractal layout for an [M, N] FP32 accumulator.

    Verified against A5 case4 l0c.wr_log.dump: layout is
        [N/16, M/16, 16 elem-M, 16 elem-N] (outer N-major, inner cell row-major)
    i.e. emit C[bm*16+x, bn*16+y] in order bn, bm, x, y. FP32 has C0=16
    elements; cell is [16, 16] floats = 1024B. Implemented as a vectorized
    block-axis transpose (no Python loops).
    """
    assert valid_m % 16 == 0 and valid_n % 16 == 0
    c = np.frombuffer(c_fp32_bytes, dtype=np.float32).reshape(valid_m // 16, 16, valid_n // 16, 16)
    # c is [bm, x, bn, y]; target order is [bn, bm, x, y].
    return c.transpose(2, 0, 1, 3).tobytes()


@dataclass(frozen=True)
class CaseGeometry:
    """Geometry + FP4 byte buffers for one matmul case (NZ/ZN/L0C helpers)."""

    valid_m: int
    valid_k: int
    valid_n: int
    a_nd_bytes: bytes
    b_nd_bytes: bytes
    c_fp32_bytes: bytes | None = None


def _write_file(path, data):
    with open(path, "wb") as f:
        f.write(data)


def write_expected_nz(geom, out_dir):
    """Write the NZ/ZN/L0C fractal-layout artifacts for sim-log comparison."""
    a_nz = nd_to_nz(geom.a_nd_bytes, rows=geom.valid_m, cols=geom.valid_k)
    b_nz = nd_to_nz(geom.b_nd_bytes, rows=geom.valid_k, cols=geom.valid_n)
    b_zn = nd_to_zn(geom.b_nd_bytes, rows=geom.valid_k, cols=geom.valid_n)
    _write_file(os.path.join(out_dir, "a_data.expected_l1_nz.bin"), a_nz)
    _write_file(os.path.join(out_dir, "b_data.expected_l1_nz.bin"), b_nz)
    # L0A reuses L1's NZ layout (TEXTRACT L1->L0A is a 1:1 read); L0B uses ZN.
    _write_file(os.path.join(out_dir, "a_data.expected_l0a_nz.bin"), a_nz)
    _write_file(os.path.join(out_dir, "b_data.expected_l0b_zn.bin"), b_zn)
    if geom.c_fp32_bytes is not None:
        # L0C FP32 fractal (post-MMAD_MX accumulator); fixpipe casts it to ND on GM store.
        c_l0c = fp32_nd_to_l0c(geom.c_fp32_bytes, geom.valid_m, geom.valid_n)
        _write_file(os.path.join(out_dir, "c_data.expected_l0c_fp32.bin"), c_l0c)
    return len(a_nz), len(b_nz)


def gen_case(valid_m, valid_k, valid_n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    a_bf16 = make_bf16_matrix(valid_m, valid_k)  # groups along K (row-major ravel)
    b_bf16 = make_bf16_matrix(valid_k, valid_n, group_axis="col")  # groups along K
    a_fp4_data, a_scale = quantize_to_hif4_a(a_bf16)
    b_fp4_data, b_scale = quantize_to_hif4_b(b_bf16)
    golden = reference_matmul(a_bf16, b_bf16)
    golden_fp32 = reference_matmul_fp32(a_bf16, b_bf16)

    with open(os.path.join(out_dir, "a_data.bin"), "wb") as f:
        f.write(a_fp4_data)
    with open(os.path.join(out_dir, "a_scale.bin"), "wb") as f:
        f.write(a_scale)
    with open(os.path.join(out_dir, "b_data.bin"), "wb") as f:
        f.write(b_fp4_data)
    with open(os.path.join(out_dir, "b_scale.bin"), "wb") as f:
        f.write(b_scale)
    with open(os.path.join(out_dir, "golden_out.bin"), "wb") as f:
        f.write(golden.tobytes())
    with open(os.path.join(out_dir, "golden.bin"), "wb") as f:
        f.write(golden.tobytes())

    geom = CaseGeometry(valid_m, valid_k, valid_n, a_fp4_data, b_fp4_data, golden_fp32.tobytes())
    a_nz_n, b_nz_n = write_expected_nz(geom, out_dir)

    print(
        f"[{os.path.basename(out_dir)}] M={valid_m} K={valid_k} N={valid_n}: "
        f"a_data={len(a_fp4_data)}B a_scale={len(a_scale)}B "
        f"b_data={len(b_fp4_data)}B b_scale={len(b_scale)}B "
        f"golden={golden.nbytes}B "
        f"a_nz={a_nz_n}B b_nz={b_nz_n}B"
    )


DEFAULT_CASES = [
    ("TMATMUL_MX_HIF4_A6_TEST.case_hif4_128x128x128_nd", 128, 128, 128),
    ("TMATMUL_MX_HIF4_A6_TEST.case_hif4_128x256x128_nd", 128, 256, 128),
    ("TMATMUL_MX_HIF4_A6_TEST.case_hif4_256x128x128_nd", 256, 128, 128),
    ("TMATMUL_MX_HIF4_A6_TEST.case_hif4_64x64x64_nd", 64, 64, 64),
    ("TMATMUL_MX_HIF4_A6_TEST.case_hif4_256x256x256_nd", 256, 256, 256),
    ("TMATMUL_MX_HIF4_A6_TEST.case_hif4_128x512x128_nd", 128, 512, 128),
    ("TMATMUL_MX_HIF4_A6_TEST.case_hif4_512x128x512_nd", 512, 128, 512),
    ("TMATMUL_MX_HIF4_A6_TEST.case_hif4_128x128x256_nd", 128, 128, 256),
    ("TMATMUL_MX_HIF4_A6_TEST.case_hif4_256x128x512_nd", 256, 128, 512),
]

CASE_PARAMS = [Hif4MatmulParams(m, k, n) for (_, m, k, n) in DEFAULT_CASES]
DEFAULT_PARAMS = CASE_PARAMS[0]

CASE_NAME = DEFAULT_PARAMS.case_name
VALID_M = DEFAULT_PARAMS.valid_m
VALID_K = DEFAULT_PARAMS.valid_k
VALID_N = DEFAULT_PARAMS.valid_n
ARTIFACTS = DEFAULT_PARAMS.artifacts


def main():
    parser = argparse.ArgumentParser(description="Generate HiF4 matmul golden data.")
    parser.add_argument("--case", type=int, default=-1)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for i, (name, m, k, n) in enumerate(DEFAULT_CASES):
            print(f"  [{i}] {name}: M={m} K={k} N={n}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.case < 0:
        for name, m, k, n in DEFAULT_CASES:
            gen_case(m, k, n, os.path.join(script_dir, name))
        _, first_m, first_k, first_n = DEFAULT_CASES[0]
        gen_case(first_m, first_k, first_n, script_dir)
    else:
        if args.case >= len(DEFAULT_CASES):
            parser.error(f"--case {args.case} out of range (0..{len(DEFAULT_CASES) - 1})")
        case_name, m, k, n = DEFAULT_CASES[args.case]
        gen_case(m, k, n, os.path.join(script_dir, case_name))
        gen_case(m, k, n, script_dir)


if __name__ == "__main__":
    main()
