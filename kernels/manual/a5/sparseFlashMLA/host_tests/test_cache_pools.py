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

"""Exercise physical cache ownership using small fixtures and actual references."""

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
from reference import TOKEN_BYTES, bf16_bits, sparse_attention


class CachePoolTests(unittest.TestCase):
    def generate(self, directory, **overrides):
        options = {
            "batch_size": 2,
            "query_length": 2,
            "num_heads": 64,
            "top_k": 17,
            "kv_length": 11,
            "page_size": 4,
            "tile_n": 64,
        }
        options.update(overrides)
        argv = ["--output", str(directory)]
        for key, value in options.items():
            if value is not None and value is not False:
                argv.append("--" + key.replace("_", "-"))
                if value is not True:
                    argv.append(str(value))
        with redirect_stdout(io.StringIO()):
            manifest = gen_data.generate(gen_data.parser().parse_args(argv))
        self.assertEqual(manifest, json.loads((Path(directory) / "fixture.json").read_text()))
        arrays = {
            name: np.fromfile(Path(directory) / name, dtype=spec["dtype"]).reshape(spec["shape"])
            for name, spec in manifest["files"].items()
        }
        return manifest, arrays

    def assert_pool(self, manifest, arrays, *, mode, logical_length, page_size, extra=False):
        prefix = "extra_" if extra else ""
        count = manifest["batch_size"] if mode == "per-batch" else 1
        pages = (logical_length + page_size - 1) // page_size
        stride = pages * page_size
        self.assertEqual(
            manifest[prefix + "kv_pool"],
            {
                "mode": mode,
                "count": count,
                "logical_length_per_pool": logical_length,
                "pages_per_pool": pages,
                "page_size": page_size,
                "allocated_tokens_per_pool": stride,
                "num_pages": count * pages,
            },
        )
        self.assertEqual(manifest[prefix + "num_pages"], count * pages)
        cache = arrays[prefix + "kv_cache.bin"]
        self.assertEqual(cache.shape, (count * pages, page_size, TOKEN_BYTES))
        for batch, queries in enumerate(arrays[prefix + "indices.bin"]):
            offset = batch * stride if mode == "per-batch" else 0
            # Sentinels must survive offsetting, including batches after zero.
            self.assertTrue(np.all((queries == -1) | (queries >= offset)))
            valid = queries[queries >= 0]
            self.assertGreater(valid.size, 0)
            self.assertTrue(np.all(valid < offset + logical_length))
        if count > 1:
            for batch in range(1, count):
                self.assertFalse(np.array_equal(cache[:pages], cache[batch * pages : (batch + 1) * pages]))

    def assert_batch_references(self, manifest, arrays):
        # Rebase each batch's physical indices onto its own cache slice. This
        # validates that generated references use each batch's actual payload.
        for batch in range(manifest["batch_size"]):
            caches, indices = {}, {}
            for prefix in ("", "extra_"):
                if prefix + "kv_cache.bin" not in arrays:
                    continue
                pool = manifest[prefix + "kv_pool"]
                owner = batch if pool["mode"] == "per-batch" else 0
                first = owner * pool["pages_per_pool"]
                caches[prefix] = arrays[prefix + "kv_cache.bin"][first : first + pool["pages_per_pool"]]
                physical = arrays[prefix + "indices.bin"][batch : batch + 1]
                indices[prefix] = np.where(
                    physical >= 0, physical - owner * pool["allocated_tokens_per_pool"], -1
                ).astype(np.int32)
            options = {"softmax_scale": manifest["softmax_scale"], "tile_n": manifest["tile_n"]}
            for filename, option in (
                ("topk_length.bin", "topk_length"),
                ("extra_topk_length.bin", "extra_topk_length"),
            ):
                if filename in arrays:
                    options[option] = arrays[filename][batch : batch + 1]
            if "attn_sink.bin" in arrays:
                options["sink"] = arrays["attn_sink.bin"]
            if "extra_" in caches:
                options.update(extra_kv_cache=caches["extra_"], extra_indices=indices["extra_"])
            for device, output_file, lse_file in (
                (False, "golden.bin", "golden_lse.bin"),
                (True, "device_ref.bin", "device_ref_lse.bin"),
            ):
                output, lse = sparse_attention(
                    arrays["q.bin"][batch : batch + 1], caches[""], indices[""], device=device, **options
                )
                np.testing.assert_array_equal(bf16_bits(output), arrays[output_file][batch : batch + 1])
                np.testing.assert_array_equal(lse, arrays[lse_file][batch : batch + 1])

    def test_default_per_batch_pools_have_disjoint_ranges_and_independent_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, arrays = self.generate(directory, batch_size=3, lengths="17,0,9", sink=True)
            self.assert_pool(manifest, arrays, mode="per-batch", logical_length=11, page_size=4)
            self.assertTrue(np.any(arrays["indices.bin"] == -1))
            np.testing.assert_array_equal(arrays["device_ref.bin"][1], 0)
            self.assert_batch_references(manifest, arrays)

    def test_offsets_round_each_batch_to_whole_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            for length in (3, 4, 5):
                with self.subTest(kv_length=length):
                    manifest, arrays = self.generate(directory, kv_length=length)
                    self.assert_pool(manifest, arrays, mode="per-batch", logical_length=length, page_size=4)

    def test_invalid_first_tile_remains_minus_one_in_every_batch_and_query(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, arrays = self.generate(directory, top_k=81, kv_length=5, first_tile_invalid=True)
            self.assert_pool(manifest, arrays, mode="per-batch", logical_length=5, page_size=4)
            np.testing.assert_array_equal(arrays["indices.bin"][..., :64], -1)
            self.assert_batch_references(manifest, arrays)

    def test_extra_cache_follows_pool_mode_and_its_own_page_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("per-batch", "shared"):
                with self.subTest(mode=mode):
                    manifest, arrays = self.generate(
                        directory, kv_pool=mode, extra_top_k=7, extra_page_size=3, no_topk_length=True
                    )
                    self.assert_pool(manifest, arrays, mode=mode, logical_length=11, page_size=4)
                    self.assert_pool(manifest, arrays, mode=mode, logical_length=7, page_size=3, extra=True)
                    self.assertNotIn("topk_length.bin", arrays)
                    self.assert_batch_references(manifest, arrays)

    def test_more_queries_do_not_allocate_more_cache_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("per-batch", "shared"):
                with self.subTest(mode=mode):
                    first, a = self.generate(
                        Path(directory) / "one", kv_pool=mode, query_length=1, extra_top_k=7, extra_page_size=3
                    )
                    more, b = self.generate(
                        Path(directory) / "three", kv_pool=mode, query_length=3, extra_top_k=7, extra_page_size=3
                    )
                    for prefix in ("", "extra_"):
                        self.assertEqual(first[prefix + "kv_pool"], more[prefix + "kv_pool"])
                        self.assertEqual(a[prefix + "kv_cache.bin"].shape, b[prefix + "kv_cache.bin"].shape)
                    self.assertEqual(b["indices.bin"].shape, (2, 3, 17))

    def test_controlled_rates_round_globally_for_both_pool_modes_and_h128(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("per-batch", "shared"):
                for rate, events in ((12.5, 1), (37.5, 2)):
                    with self.subTest(mode=mode, rate=rate):
                        manifest, arrays = self.generate(
                            directory,
                            kv_pool=mode,
                            query_length=1,
                            num_heads=128,
                            top_k=65,
                            kv_length=67,
                            page_size=17,
                            rescale_rate=rate,
                        )
                        self.assert_pool(manifest, arrays, mode=mode, logical_length=67, page_size=17)
                        stats = {}
                        sparse_attention(
                            arrays["q.bin"],
                            arrays["kv_cache.bin"],
                            arrays["indices.bin"],
                            topk_length=arrays["topk_length.bin"],
                            softmax_scale=manifest["softmax_scale"],
                            tile_n=64,
                            device=True,
                            rescale_stats=stats,
                        )
                        self.assertEqual(stats["eligible_decisions"], 4)
                        self.assertEqual(stats["conditional_flushes"], events)
                        self.assertEqual(len(stats["per_job"]), 4)
                        self.assertEqual(manifest["rescale_stats"]["requested_rate_percent"], rate)
                        for key in ("eligible_decisions", "conditional_flushes", "rate_percent", "per_job"):
                            self.assertEqual(manifest["rescale_stats"][key], stats[key])
                        self.assert_batch_references(manifest, arrays)

    def test_reuse_inputs_preserves_pool_metadata_and_input_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("per-batch", "shared"):
                with self.subTest(mode=mode):
                    original, arrays = self.generate(
                        Path(directory) / "source", kv_pool=mode, extra_top_k=7, extra_page_size=3
                    )
                    reused, copied = self.generate(
                        Path(directory) / "reused", reuse_inputs=Path(directory) / "source", tile_n=128
                    )
                    for key in ("kv_pool", "extra_kv_pool", "num_pages", "extra_num_pages"):
                        self.assertEqual(original[key], reused[key])
                    for name, values in arrays.items():
                        if name not in ("golden.bin", "golden_lse.bin", "device_ref.bin", "device_ref_lse.bin"):
                            np.testing.assert_array_equal(values, copied[name])
                    self.assertEqual(reused["tile_n"], 128)
                    self.assert_batch_references(reused, copied)


if __name__ == "__main__":
    unittest.main()
