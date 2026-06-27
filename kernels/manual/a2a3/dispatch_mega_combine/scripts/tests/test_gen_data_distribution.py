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

import argparse
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

import gen_data


def make_args(world_size: int, m: int, topk: int = 8, experts: int = 16) -> argparse.Namespace:
    return argparse.Namespace(
        world_size=world_size,
        m=m,
        k=128,
        n=128,
        topk=topk,
        experts=experts,
        max_output_size=81940,
        aic_num=24,
        aiv_num=48,
        case_mode="cpu-golden",
        golden_backend="python-batch",
        golden_chunk_rows=512,
        seed=20260515,
        atol=1e-4,
        rtol=1e-3,
        reuse_data=True,
    )


def route_counts(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    expert_counts: Counter[int] = Counter()
    dst_rank_counts = [0 for _ in range(args.world_size)]
    for rank in range(args.world_size):
        expert_idx = gen_data.make_expert_idx(rank, args)
        for expert in expert_idx.reshape(-1):
            expert = int(expert)
            expert_counts[expert] += 1
            dst_rank_counts[expert // args.experts] += 1
    total_experts = args.world_size * args.experts
    return [expert_counts[i] for i in range(total_experts)], dst_rank_counts


class GenDataDistributionTest(unittest.TestCase):
    def test_global_token_round_robin_balances_16_rank_m16_case(self) -> None:
        args = make_args(world_size=16, m=16)

        expert_counts, dst_rank_counts = route_counts(args)

        self.assertEqual(set(expert_counts), {8})
        self.assertEqual(dst_rank_counts, [128] * 16)

    def test_global_token_round_robin_balances_8_and_16_rank_multiples_of_16(self) -> None:
        for world_size, m in ((8, 16), (8, 32), (16, 16), (16, 32), (16, 64)):
            with self.subTest(world_size=world_size, m=m):
                args = make_args(world_size=world_size, m=m)

                expert_counts, dst_rank_counts = route_counts(args)

                self.assertEqual(len(set(expert_counts)), 1)
                self.assertEqual(len(set(dst_rank_counts)), 1)

    def test_reuse_data_rejects_missing_cache_version(self) -> None:
        args = make_args(world_size=2, m=16, topk=2, experts=2)
        metadata = gen_data.build_case_metadata(args)
        metadata.pop("data_cache_version")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            (out_dir / "case.json").write_text(json.dumps(metadata), encoding="utf-8")
            sizes = gen_data.expected_rank_file_sizes(args)
            for rank in range(args.world_size):
                for name, size in sizes.items():
                    np.zeros(size, dtype=np.uint8).tofile(out_dir / f"rank{rank}_{name}.bin")

            mismatch = gen_data.reusable_data_mismatch(out_dir, args)

        self.assertIn("data_cache_version", mismatch)


if __name__ == "__main__":
    unittest.main()
