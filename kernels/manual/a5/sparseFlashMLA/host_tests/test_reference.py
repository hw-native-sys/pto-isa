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

"""Independent format, sparse attention, rescaling, and fixture checks."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "golden"))
import gen_data
import gen_validation
from reference import (
    DIM,
    TOKEN_BYTES,
    bf16,
    bf16_bits,
    decode_bf16,
    dequantize_fp4_cache,
    quantize_fp4_cache,
    sparse_attention,
)


def query(batch=1, seq_q=1, heads=64):
    return np.zeros((batch, seq_q, heads, DIM), dtype=np.uint16)


def constant_cache(values, page_size=1):
    values = np.asarray(values, dtype=np.float32)
    pages = (len(values) + page_size - 1) // page_size
    raw = np.zeros((pages * page_size, DIM), dtype=np.float32)
    raw[: len(values)] = values[:, None]
    return gen_validation.encode_exact(raw).reshape(pages, page_size, TOKEN_BYTES)


def load_arrays(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "fixture.json").read_text())
    arrays = {
        name: np.fromfile(directory / name, dtype=spec["dtype"]).reshape(spec["shape"])
        for name, spec in metadata["files"].items()
    }
    return metadata, arrays


class FormatTests(unittest.TestCase):
    def test_bf16_storage_and_dtype_validation(self):
        values = np.array([0, 1, -3, 1.00390625, 2688], dtype=np.float32)
        self.assertEqual(bf16_bits(values).dtype, np.uint16)
        np.testing.assert_array_equal(decode_bf16(bf16_bits(values)), bf16(values))
        with self.assertRaises(ValueError):
            decode_bf16(np.zeros(4, dtype=np.uint8))

    def test_low_even_nibbles_and_scale_groups(self):
        record = np.zeros((1, 1, TOKEN_BYTES), dtype=np.uint8)
        record[..., :256] = 0xA1
        scales = np.exp2(np.arange(32) % 12 - 9).astype(ml_dtypes.float8_e4m3fn)
        record[..., 256:] = scales.view(np.uint8)
        decoded = dequantize_fp4_cache(record)[0, 0]
        np.testing.assert_array_equal(decoded[::16], scales.astype(np.float32) * 0.5)
        np.testing.assert_array_equal(decoded[1::16], -scales.astype(np.float32))

    def test_every_finite_e4m3_scale_and_fp4_code_is_exact_in_bf16(self):
        records = np.zeros((8, TOKEN_BYTES), dtype=np.uint8)
        codes = np.tile(np.arange(16, dtype=np.uint8), (8, 32))
        records[:, :256] = codes[:, ::2] | (codes[:, 1::2] << 4)
        records[:, 256:] = np.resize([v for v in range(256) if v & 127 != 127], (8, 32))
        decoded = dequantize_fp4_cache(records)
        np.testing.assert_array_equal(decoded, bf16(decoded))
        self.assertEqual(decoded.max(), 2688)
        self.assertEqual(decoded.min(), -2688)

    def test_quantization_rne_ties_and_saturation(self):
        values = np.zeros((1, DIM), dtype=np.float32)
        # The first element fixes scale=1. Remaining values land at midpoints.
        values[0, :8] = [6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]
        packed = quantize_fp4_cache(values)
        decoded = dequantize_fp4_cache(packed)
        np.testing.assert_array_equal(decoded[0, :8], [6, 0, 1, 1, 2, 2, 4, 4])
        values.fill(1e8)
        self.assertEqual(float(dequantize_fp4_cache(quantize_fp4_cache(values)).max()), 2688)

    def test_invalid_scale_and_record_shape_rejected(self):
        record = constant_cache([1])
        record[..., 256] = 0x7F
        with self.assertRaises(ValueError):
            dequantize_fp4_cache(record)
        with self.assertRaises(ValueError):
            dequantize_fp4_cache(np.zeros((1, 528), dtype=np.uint8))


class AttentionTests(unittest.TestCase):
    def test_physical_indices_duplicates_and_page_tails(self):
        cache = constant_cache([1, 2, 3, 4, 6], 3)
        index = np.array([[[4, 0, 4, -1, 2]]], dtype=np.int32)
        for tile_n in (64, 128):
            for device in (False, True):
                out, lse = sparse_attention(query(), cache, index, tile_n=tile_n, device=device)
                np.testing.assert_array_equal(out, bf16((6 + 1 + 6 + 3) / 4))
                np.testing.assert_allclose(lse, np.log(4), atol=1e-6, rtol=0)

    def test_prefix_tail_is_never_gathered_or_validated(self):
        index = np.array([[[1, -1, 99999, -12345]]], dtype=np.int32)
        for device in (False, True):
            out, lse = sparse_attention(
                query(), constant_cache([1, 3]), index, topk_length=np.array([2], dtype=np.int32), device=device
            )
            np.testing.assert_array_equal(out, 3)
            np.testing.assert_array_equal(lse, 0)
        for bad in (-2, 2):
            with self.assertRaises(ValueError):
                sparse_attention(query(), constant_cache([1, 3]), np.array([[[bad]]], dtype=np.int32))

    def test_lengths_at_both_tile_boundaries_and_persistent_rows(self):
        lengths = np.array([0, 1, 63, 64, 65, 127, 128, 129, 257], dtype=np.int32)
        q = query(batch=len(lengths), seq_q=2, heads=128)
        index = np.zeros((len(lengths), 2, 257), dtype=np.int32)
        index[..., 1::2] = 1
        for tile_n in (64, 128):
            out, lse = sparse_attention(
                q, constant_cache([1, 3]), index, topk_length=lengths, tile_n=tile_n, device=True
            )
            np.testing.assert_array_equal(out[0], 0)
            self.assertTrue(np.all(np.isposinf(lse[0])))
            for b, length in enumerate(lengths[1:], 1):
                np.testing.assert_array_equal(out[b], bf16((length + 2 * (length // 2)) / length))
                np.testing.assert_allclose(lse[b], np.log(length), atol=1e-6, rtol=0)

    def test_main_extra_scopes_use_one_joint_denominator(self):
        q = query()
        q[..., 0] = bf16_bits(1)
        index = np.zeros((1, 1, 1), dtype=np.int32)
        for tile_n in (64, 128):
            for device in (False, True):
                out, lse = sparse_attention(
                    q,
                    constant_cache([1]),
                    index,
                    extra_kv_cache=constant_cache([3]),
                    extra_indices=index,
                    softmax_scale=1,
                    tile_n=tile_n,
                    device=device,
                )
                self.assertLess(np.abs(out[0, 0, 0, 0] - (np.exp(-2) + 3) / (np.exp(-2) + 1)), 0.02)
                np.testing.assert_allclose(lse, np.logaddexp(1, 3), atol=1e-6, rtol=0)

    def test_empty_primary_and_valid_extra(self):
        for device in (False, True):
            out, lse = sparse_attention(
                query(),
                constant_cache([1]),
                np.full((1, 1, 129), -1, dtype=np.int32),
                extra_kv_cache=constant_cache([4]),
                extra_indices=np.zeros((1, 1, 3), dtype=np.int32),
                device=device,
            )
            np.testing.assert_array_equal(out, 4)
            np.testing.assert_allclose(lse, np.log(3), atol=1e-6, rtol=0)

    def test_sink_normalization_and_returned_lse(self):
        sink = np.zeros(64, dtype=np.float32)
        sink[:3] = [np.inf, -np.inf, 0]
        for device in (False, True):
            out, lse = sparse_attention(
                query(), constant_cache([4]), np.zeros((1, 1, 1), dtype=np.int32), sink=sink, device=device
            )
            np.testing.assert_array_equal(out[0, 0, :3, 0], [0, 4, 2])
            np.testing.assert_array_equal(lse, 0)

    def test_all_empty_variants_are_zero_with_positive_infinite_lse(self):
        for width in (0, 129):
            for device in (False, True):
                for sink in (None, np.full(64, np.inf), np.full(64, -np.inf), np.zeros(64)):
                    out, lse = sparse_attention(
                        query(),
                        constant_cache([1]),
                        np.full((1, 1, width), -1, dtype=np.int32),
                        sink=sink,
                        device=device,
                    )
                    np.testing.assert_array_equal(out, 0)
                    self.assertTrue(np.all(np.isposinf(lse)))

    def test_masked_tile_between_valid_tiles(self):
        for tile_n in (64, 128):
            indices = np.zeros((1, 1, 3 * tile_n), dtype=np.int32)
            indices[..., tile_n : 2 * tile_n] = -1
            indices[..., 2 * tile_n :] = 1
            out, lse = sparse_attention(query(), constant_cache([1, 3]), indices, tile_n=tile_n, device=True)
            np.testing.assert_array_equal(out, 2)
            np.testing.assert_allclose(lse, np.log(2 * tile_n), atol=1e-6, rtol=0)

    def test_reject_invalid_attention_options(self):
        args = (query(), constant_cache([1]), np.zeros((1, 1, 1), dtype=np.int32))
        for options in (
            {"tile_n": 256},
            {"p_scale": 256},
            {"topk_length": np.array([-1], np.int32)},
            {"topk_length": np.array([2], np.int32)},
            {"extra_indices": args[2]},
            {"sink": np.full(64, np.nan)},
            {"softmax_scale": np.inf},
        ):
            with self.assertRaises(ValueError):
                sparse_attention(*args, **options)


class FixtureTests(unittest.TestCase):
    def test_binary_schema_and_default_tile(self):
        with tempfile.TemporaryDirectory() as base:
            manifest = gen_data.generate(
                gen_data.parser().parse_args(
                    [
                        "--output",
                        base,
                        "--batch",
                        "2",
                        "--sq",
                        "2",
                        "--topk",
                        "65",
                        "--lengths",
                        "0,65",
                        "--extra-top-k",
                        "3",
                        "--sink",
                    ]
                )
            )
            self.assertEqual(manifest["tile_n"], 128)
            self.assertEqual(manifest["probability_scale"], 1)
            self.assertEqual(manifest["query_format"], "bf16")
            self.assertEqual(manifest["pv_format"], "bf16")
            self.assertEqual(manifest["files"]["q.bin"]["dtype"], "uint16")
            self.assertNotIn("q_scale.bin", manifest["files"])
            for name, spec in manifest["files"].items():
                self.assertEqual(
                    (Path(base) / name).stat().st_size, np.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize
                )

    def test_same_inputs_and_exact_math_for_n64_n128(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "n64", Path(directory) / "n128"
            original = gen_data.generate(
                gen_data.parser().parse_args(
                    [
                        "--output",
                        str(first),
                        "--tile-n",
                        "64",
                        "--top-k",
                        "257",
                        "--extra-top-k",
                        "33",
                        "--page-size",
                        "37",
                        "--sink",
                    ]
                )
            )
            gen_data.reuse_inputs(first, second, 128)
            for name in original["files"]:
                if name not in ("device_ref.bin", "device_ref_lse.bin"):
                    self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def test_reuse_rejects_missing_format_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            original = gen_data.generate(gen_data.parser().parse_args(["--output", directory, "--top-k", "1"]))
            for key in ("query_format", "pv_format", "cache_format"):
                modified = dict(original)
                modified.pop(key)
                (Path(directory) / "fixture.json").write_text(json.dumps(modified))
                with self.assertRaises(ValueError):
                    gen_data.reuse_inputs(directory, Path(directory) / "other", 128)

    def test_delayed_rescaling_matches_independent_closed_form(self):
        for tile_n in (64, 128):
            with tempfile.TemporaryDirectory() as directory:
                case = gen_validation.generate_case("rescale", directory, tile_n)
                metadata, arrays = load_arrays(directory)
                self.assertEqual(case["nontrivial_rescale_head_updates"], 64)
                self.assertEqual(metadata["nontrivial_rescale_head_updates"], 64)
                decoded = dequantize_fp4_cache(arrays["kv_cache.bin"]).reshape(-1, DIM)[arrays["indices.bin"][0, 0]]
                scores = decoded[:, 0] + 0.5 * decoded[:, 1]
                maxima = scores.reshape(4, tile_n).max(axis=-1).astype(np.float64)
                np.testing.assert_allclose(np.diff(maxima), 2, atol=0, rtol=0)
                offsets = scores[:tile_n].astype(np.float64) - maxima[0]
                local = np.exp(offsets)
                weights = np.exp(maxima - maxima[-1])
                denominator = local.sum() * weights.sum()
                rounded_sum = bf16(local).astype(np.float64).sum()
                # First three tiles retain the first anchor, then the six-logit
                # jump flushes their pending segment before the fourth tile.
                pending = sum(
                    bf16(np.exp(offsets + shift)).astype(np.float64).sum() * value
                    for shift, value in ((0, 1), (2, 2), (4, 3))
                )
                expected = (pending * np.exp(-6) + rounded_sum * 4) / denominator
                actual = decode_bf16(arrays["device_ref.bin"])[0, 0, :, 16]
                np.testing.assert_array_equal(actual, bf16(expected))
                np.testing.assert_allclose(
                    arrays["device_ref_lse.bin"], np.log(denominator) + maxima[-1], atol=2e-6, rtol=0
                )
                # Losing alpha in either state yields a visibly different output.
                no_alpha_either = rounded_sum * 10 / (local.sum() * 4)
                no_alpha_numerator = rounded_sum * 10 / denominator
                self.assertGreater(float(np.min(np.abs(actual - no_alpha_either))), 0.5)
                self.assertGreater(float(np.min(np.abs(actual - no_alpha_numerator))), 0.5)

    def test_adversarial_cache_has_no_fp8_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            gen_validation.generate_case("adversarial_cache", directory)
            _, arrays = load_arrays(directory)
            decoded = dequantize_fp4_cache(arrays["kv_cache.bin"]).reshape(-1, DIM)[:8]
            self.assertEqual(np.abs(decoded).max(), 2688)
            expected = bf16(decoded.astype(np.float64).mean(axis=0))
            actual = decode_bf16(arrays["device_ref.bin"])[0, 0]
            np.testing.assert_array_equal(actual, np.broadcast_to(expected, actual.shape))
            clipped_mean = bf16(np.clip(decoded, -448, 448).astype(np.float64).mean(axis=0))
            self.assertGreater(np.max(np.abs(actual[0] - clipped_mean)), 100)

    def test_validation_suite_has_masks_empty_jobs_and_rescale(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = gen_validation.generate_suite(directory)
            self.assertGreater(suite["total_tiles"], 0)
            _, arrays = load_arrays(Path(directory) / "persistent")
            np.testing.assert_array_equal(arrays["device_ref.bin"][0], 0)
            self.assertTrue(np.all(np.isposinf(arrays["device_ref_lse.bin"][0])))
            witness = next(case for case in suite["cases"] if case["name"] == "rescale")
            self.assertGreater(witness["nontrivial_rescale_head_updates"], 0)

    def test_job_count_fixtures_cover_empty_jobs_and_large_values(self):
        with tempfile.TemporaryDirectory() as directory:
            for jobs in (1, 2, 3, 5):
                base = Path(directory) / f"jobs{jobs}"
                gen_validation.generate_case(f"jobs{jobs}", base)
                metadata, arrays = load_arrays(base)
                self.assertEqual(metadata["batch_size"], jobs)
                self.assertEqual(metadata["query_length"] * metadata["num_heads"] // 64, 1)
                if jobs > 1:
                    empty = (arrays["topk_length.bin"] == 0) & (arrays["extra_topk_length.bin"] == 0)
                    self.assertTrue(np.any(empty))
                    np.testing.assert_array_equal(arrays["device_ref.bin"][empty], 0)
                    self.assertTrue(np.all(np.isposinf(arrays["device_ref_lse.bin"][empty])))
            base = Path(directory) / "large"
            gen_validation.generate_case("large_cache", base)
            _, arrays = load_arrays(base)
            np.testing.assert_array_equal(decode_bf16(arrays["device_ref.bin"]), 2688)
            np.testing.assert_array_equal(arrays["device_ref_lse.bin"], 0)

    def test_two_job_benchmark_has_distinct_states_and_rescales_both_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            case = gen_validation.generate_case("topk1024_jobs2", directory)
            metadata, arrays = load_arrays(directory)
            self.assertEqual(metadata["batch_size"], 1)
            self.assertEqual(metadata["query_length"], 2)
            self.assertEqual(metadata["top_k"], 1024)
            self.assertEqual(case["tiles"], 16)
            self.assertEqual(case["nontrivial_rescale_head_updates"], 128)
            q = decode_bf16(arrays["q.bin"])
            np.testing.assert_array_equal(q[0, :, 0, 0], [1, -1])
            np.testing.assert_array_equal(arrays["indices.bin"][0, 1], arrays["indices.bin"][0, 0, ::-1])
            output = decode_bf16(arrays["device_ref.bin"])
            np.testing.assert_array_equal(output[0, :, 0, 16], [3.84375, 1.15625])


if __name__ == "__main__":
    unittest.main()
