#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Closed-form checks of the oracle's held basis and conditional PV rescaling."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden"))
from reference import _attention, bf16, bf16_bits


def run_tiles(scores, markers, *, coefficients=None, scale=1, valid=None):
    query = np.zeros((64, 512), dtype=np.float32)
    query[:, 0] = 1 if coefficients is None else coefficients
    values = np.zeros((len(scores) * 128, 512), dtype=np.float32)
    for tile, (score, marker) in enumerate(zip(scores, markers)):
        values[tile * 128 : (tile + 1) * 128, 0] = score
        values[tile * 128 : (tile + 1) * 128, 16] = marker
    mask = np.ones(len(values), dtype=bool) if valid is None else np.repeat(valid, 128)
    alpha = []
    output, lse = _attention(query, values, mask, None, True, np.float32(scale), 128, alpha)
    return bf16(output), lse, alpha


class DelayedRescaleTests(unittest.TestCase):
    def test_held_anchor_keeps_bf16_p_above_one_and_unrounded_denominator(self):
        output, lse, alpha = run_tiles([0, 2], [1, 3])
        p = np.exp(np.float32(2), dtype=np.float32)
        expected = bf16((np.float32(1) + np.float32(3) * bf16(p)) / (np.float32(1) + p))
        np.testing.assert_array_equal(output[:, 16], expected)
        np.testing.assert_array_equal(alpha[1], 1)
        np.testing.assert_allclose(lse, np.log(128 * (1 + p)), atol=1e-6, rtol=0)
        # Rounding the denominator or P after normalization changes this witness.
        self.assertNotEqual(bf16_bits(expected).item(), bf16_bits((1 + 3 * p) / (1 + p)).item())

    def test_strict_ln256_threshold_equal_skips_and_above_rescales(self):
        threshold = np.float32(np.log(256.0))
        _, _, alpha = run_tiles([0, 1, 2], [1, 2, 3], scale=threshold)
        np.testing.assert_array_equal(alpha[1], 1)
        np.testing.assert_allclose(alpha[2], np.exp(-2 * threshold), atol=0, rtol=1e-6)
        self.assertTrue(np.all(alpha[2] < 1))

    def test_only_one_aiv_requests_rescale_but_all_its_rows_change_anchor(self):
        coefficients = np.concatenate([np.ones(16), np.full(16, 0.5), np.full(32, -1)]).astype(np.float32)
        output, _, alpha = run_tiles([0, 8], [1, 3], coefficients=coefficients)
        np.testing.assert_allclose(alpha[1][:16], np.exp(np.float32(-8)), atol=0, rtol=1e-6)
        np.testing.assert_allclose(alpha[1][16:32], np.exp(np.float32(-4)), atol=0, rtol=1e-6)
        np.testing.assert_array_equal(alpha[1][32:], 1)
        p = np.exp(np.float32(-8), dtype=np.float32)
        expected = bf16((1 + 3 * bf16(p)) / (1 + p))
        np.testing.assert_array_equal(output[32:, 16], expected)

    def test_huge_jump_updates_anchor_before_exp_after_masked_prefix(self):
        output, lse, alpha = run_tiles([0, 128], [1, 3], valid=[False, True])
        self.assertTrue(np.all(np.isfinite(output)))
        np.testing.assert_array_equal(output[:, 16], 3)
        np.testing.assert_array_equal(alpha[0], 0)
        np.testing.assert_array_equal(alpha[1], 0)
        np.testing.assert_allclose(lse, 128 + np.log(128), atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
