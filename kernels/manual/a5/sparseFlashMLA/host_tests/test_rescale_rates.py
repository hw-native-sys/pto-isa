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

"""Host checks of actual conditional flush counts and controlled-rate fixtures."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden"))
import gen_data
from gen_validation import encode_exact
from reference import DIM, TOKEN_BYTES, _attention, bf16_bits, sparse_attention


def run_tiles(scores, *, coefficients=None, valid=None, stats=None, trace=None, tile_n=64):
    """Use exact BF16 scores so predicate expectations need no numerical oracle."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores[:, None]
    heads = 64 if coefficients is None else len(coefficients)
    query = np.zeros((heads, DIM), dtype=np.float32)
    query[:, : scores.shape[1]] = 1 if coefficients is None else coefficients
    values = np.zeros((len(scores) * tile_n, DIM), dtype=np.float32)
    values[:, : scores.shape[1]] = scores.repeat(tile_n, axis=0)
    values[:, 16] = np.repeat(np.arange(1, len(scores) + 1), tile_n)
    mask = np.ones(len(values), dtype=bool) if valid is None else np.repeat(valid, tile_n)
    return _attention(query, values, mask, None, True, np.float32(1), tile_n, trace, stats)


class RescaleStatisticsTests(unittest.TestCase):
    def test_counts_boolean_trigger_when_alpha_underflows_or_prefix_is_invalid(self):
        # Alpha=0 on tile 0, on a masked prefix, and after a large jump. Only
        # actual post-initial predicate publications count toward the rate.
        for valid, expected in (([True, True, True], [2]), ([False, False, True], [2])):
            with self.subTest(valid=valid):
                stats, trace = {}, []
                output, lse = run_tiles([0, 0, 128], valid=valid, stats=stats, trace=trace)
                np.testing.assert_array_equal(trace[0], 0)
                np.testing.assert_array_equal(trace[2], 0)
                if not valid[0]:
                    np.testing.assert_array_equal(trace[1], 0)
                self.assertEqual(stats["eligible_decisions"], 2)
                self.assertEqual(stats["conditional_flushes"], 1)
                self.assertEqual(stats["rate_percent"], 50)
                self.assertEqual(stats["per_job"][0]["trigger_tiles"], expected)
                self.assertTrue(np.all(np.isfinite(output)))
                self.assertTrue(np.all(np.isfinite(lse)))

    def test_two_aiv_predicates_count_as_one_job_flush(self):
        for triggering_heads in ((0,), (32,), (0, 32)):
            with self.subTest(triggering_heads=triggering_heads):
                coefficients = np.zeros((64, 1), dtype=np.float32)
                coefficients[list(triggering_heads)] = 1
                stats = {}
                run_tiles([0, 8], coefficients=coefficients, stats=stats)
                self.assertEqual(stats["eligible_decisions"], 1)
                self.assertEqual(stats["conditional_flushes"], 1)
                self.assertEqual(stats["per_job"][0]["trigger_tiles"], [1])

    def test_h128_counts_each_64_head_group_independently(self):
        coefficients = np.zeros((128, 2), dtype=np.float32)
        coefficients[:64, 0], coefficients[64:, 1] = 1, 1
        stats = {}
        run_tiles([[0, 0], [8, 0], [8, 8], [16, 8]], coefficients=coefficients, stats=stats)
        self.assertEqual(stats["eligible_decisions"], 6)
        self.assertEqual(stats["conditional_flushes"], 3)
        self.assertEqual(stats["rate_percent"], 50)
        self.assertEqual([job["head_group"] for job in stats["per_job"]], [0, 1])
        self.assertEqual([job["trigger_tiles"] for job in stats["per_job"]], [[1, 3], [2]])

    def test_empty_and_all_masked_jobs_keep_length_based_denominators_and_labels(self):
        for tile_n in (64, 128):
            with self.subTest(tile_n=tile_n):
                stats = {}
                q = np.zeros((2, 2, 128, DIM), dtype=np.uint16)
                cache = encode_exact(np.zeros((1, 1, DIM), dtype=np.float32))
                indices = np.full((2, 2, 2 * tile_n + 1), -1, dtype=np.int32)
                out, lse = sparse_attention(
                    q,
                    cache,
                    indices,
                    topk_length=np.array([0, 2 * tile_n + 1], dtype=np.int32),
                    tile_n=tile_n,
                    device=True,
                    rescale_stats=stats,
                )
                np.testing.assert_array_equal(out, 0)
                self.assertTrue(np.all(np.isposinf(lse)))
                self.assertEqual(stats["eligible_decisions"], 8)
                self.assertEqual(stats["conditional_flushes"], 0)
                self.assertEqual(stats["rate_percent"], 0)
                self.assertEqual(
                    [(j["batch"], j["query"], j["head_group"]) for j in stats["per_job"]],
                    [(b, s, h) for b in range(2) for s in range(2) for h in range(2)],
                )
                self.assertEqual([j["eligible_decisions"] for j in stats["per_job"]], [0] * 4 + [2] * 4)
                self.assertTrue(all(j["trigger_tiles"] == [] for j in stats["per_job"]))

    def test_no_post_initial_tile_has_undefined_rate(self):
        for width in (0, 1, 64):
            stats = {}
            sparse_attention(
                np.zeros((1, 1, 64, DIM), dtype=np.uint16),
                encode_exact(np.zeros((1, 1, DIM), dtype=np.float32)),
                np.full((1, 1, width), -1, dtype=np.int32),
                device=True,
                tile_n=64,
                rescale_stats=stats,
            )
            self.assertEqual(stats["eligible_decisions"], 0)
            self.assertEqual(stats["conditional_flushes"], 0)
            self.assertIsNone(stats["rate_percent"])

    def test_main_and_extra_have_independent_padding_and_one_tile_sequence(self):
        for tile_n in (64, 128):
            with self.subTest(tile_n=tile_n):
                query = np.zeros((1, 1, 64, DIM), dtype=np.float32)
                query[..., 0] = 4
                values = np.zeros((3, 1, DIM), dtype=np.float32)
                values[:, 0, 0] = [0, 2, 4]
                cache = encode_exact(values).reshape(3, 1, TOKEN_BYTES)
                indices = np.zeros((1, 1, tile_n + 1), dtype=np.int32)
                indices[..., -1] = 1
                options = {
                    "extra_kv_cache": cache,
                    "extra_indices": np.full((1, 1, 1), 2, dtype=np.int32),
                    "softmax_scale": 1,
                    "tile_n": tile_n,
                    "device": True,
                }
                stats = {}
                expected = sparse_attention(bf16_bits(query), cache, indices, **options)
                actual = sparse_attention(bf16_bits(query), cache, indices, **options, rescale_stats=stats)
                for observed, unchanged in zip(actual, expected):
                    np.testing.assert_array_equal(observed, unchanged)
                self.assertEqual(stats["eligible_decisions"], 2)
                self.assertEqual(stats["conditional_flushes"], 2)
                self.assertEqual(stats["per_job"][0]["trigger_tiles"], [1, 2])

    def test_statistics_leave_numerical_output_unchanged(self):
        for valid in ([True] * 4, [False, True, True, True]):
            stats = {}
            expected = run_tiles([0, 2, 8, 10], valid=valid)
            actual = run_tiles([0, 2, 8, 10], valid=valid, stats=stats)
            for observed, unchanged in zip(actual, expected):
                np.testing.assert_array_equal(observed, unchanged)


class ControlledRescaleFixtureTests(unittest.TestCase):
    def generate_fixture(self, directory, *, tile_n, heads=64, batch=1, top_k=None, rate=None):
        top_k = 2 * tile_n + 1 if top_k is None else top_k
        argv = [
            "--output",
            str(directory),
            "--tile-n",
            str(tile_n),
            "--num-heads",
            str(heads),
            "--batch-size",
            str(batch),
            "--top-k",
            str(top_k),
            "--kv-length",
            str(max(1, top_k)),
            "--page-size",
            "17",
        ]
        if rate is not None:
            argv += ["--rescale-rate", str(rate)]
        with redirect_stdout(io.StringIO()):
            manifest = gen_data.generate(gen_data.parser().parse_args(argv))
        persisted = json.loads((Path(directory) / "fixture.json").read_text())
        self.assertEqual(manifest, persisted)
        return manifest

    def assert_recorded_stats_match_inputs(self, directory, manifest):
        arrays = {
            name: np.fromfile(Path(directory) / name, dtype=spec["dtype"]).reshape(spec["shape"])
            for name, spec in manifest["files"].items()
        }
        stats = {}
        output, lse = sparse_attention(
            arrays["q.bin"],
            arrays["kv_cache.bin"],
            arrays["indices.bin"],
            topk_length=arrays.get("topk_length.bin"),
            softmax_scale=manifest["softmax_scale"],
            tile_n=manifest["tile_n"],
            device=True,
            rescale_stats=stats,
        )
        for key in ("eligible_decisions", "conditional_flushes", "rate_percent", "per_job"):
            self.assertEqual(manifest["rescale_stats"][key], stats[key])
        np.testing.assert_array_equal(bf16_bits(output), arrays["device_ref.bin"])
        np.testing.assert_array_equal(lse, arrays["device_ref_lse.bin"])
        return stats

    def test_requested_rates_for_multiple_jobs_head_groups_and_partial_tiles(self):
        with tempfile.TemporaryDirectory() as directory:
            for tile_n in (64, 128):
                for batch, heads in ((2, 64), (1, 128)):
                    for rate in (0, 50, 100):
                        with self.subTest(tile_n=tile_n, batch=batch, heads=heads, rate=rate):
                            manifest = self.generate_fixture(
                                directory, tile_n=tile_n, batch=batch, heads=heads, rate=rate
                            )
                            stats = self.assert_recorded_stats_match_inputs(directory, manifest)
                            self.assertEqual(stats["eligible_decisions"], 4)
                            self.assertEqual(stats["conditional_flushes"], rate * 4 // 100)
                            self.assertEqual(stats["rate_percent"], rate)
                            self.assertEqual(manifest["rescale_stats"]["requested_rate_percent"], rate)
                            self.assertEqual(len(stats["per_job"]), 2)
                            for job in stats["per_job"]:
                                self.assertEqual(job["eligible_decisions"], 2)
                                self.assertTrue(all(1 <= tile <= 2 for tile in job["trigger_tiles"]))
                                self.assertEqual(job["conditional_flushes"], len(job["trigger_tiles"]))

    def test_requested_event_count_rounds_half_up_globally(self):
        with tempfile.TemporaryDirectory() as directory:
            for heads, rate, expected_rate in ((64, 50, 100), (128, 25, 50)):
                with self.subTest(heads=heads, rate=rate):
                    manifest = self.generate_fixture(directory, tile_n=64, heads=heads, top_k=65, rate=rate)
                    stats = self.assert_recorded_stats_match_inputs(directory, manifest)
                    self.assertEqual(stats["eligible_decisions"], heads // 64)
                    self.assertEqual(stats["conditional_flushes"], 1)
                    self.assertEqual(stats["rate_percent"], expected_rate)
                    self.assertEqual(manifest["rescale_stats"]["requested_rate_percent"], rate)

    def test_one_initial_tile_has_no_eligible_decision_at_any_requested_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            for rate in (0, 50, 100):
                with self.subTest(rate=rate):
                    manifest = self.generate_fixture(directory, tile_n=128, top_k=17, rate=rate)
                    stats = self.assert_recorded_stats_match_inputs(directory, manifest)
                    self.assertEqual(stats["eligible_decisions"], 0)
                    self.assertEqual(stats["conditional_flushes"], 0)
                    self.assertIsNone(stats["rate_percent"])
                    self.assertEqual(stats["per_job"][0]["trigger_tiles"], [])

    def test_default_random_generation_records_measured_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.generate_fixture(directory, tile_n=64, top_k=65)
            self.assert_recorded_stats_match_inputs(directory, manifest)
            self.assertNotIn("requested_rate_percent", manifest["rescale_stats"])


if __name__ == "__main__":
    unittest.main()
