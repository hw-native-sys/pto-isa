#!/usr/bin/python3
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

import os
import sys
import numpy as np

GATE_BIAS = np.float64(0.125)
H = 8

DEFAULT_CONFIGS = [(128, 1, 65536, "T64K"), (256, 4, 65536, "T64K"), (512, 1, 65536, "T64K"), (1024, 1, 65536, "T64K")]
VARIANTS = ["baseline", "fused"]

PERF_ANALYSIS = os.environ.get("PERF_ANALYSIS", "0") == "1"
DIMS = [128, 256, 512, 1024]
BLOCKS = [1, 4, 16, 64]
TABLES = {
    128: [65536, 262144, 1048576],
    256: [65536, 262144, 1048576],
    512: [65536, 262144, 1048576],
    1024: [65536, 262144, 1048576],
}
TABLE_TAGS = {65536: "T64K", 262144: "T256K", 1048576: "T1M"}
PATTERNS = ["RAND", "SEQ", "SAME", "STRIDE"]
PATTERN_IDX = {"RAND": 0, "SEQ": 1, "SAME": 2, "STRIDE": 3}


def gen_indices(pattern, block, table_rows, rng):
    n = block * H
    if pattern == "RAND":
        return rng.choice(table_rows, size=n, replace=True).astype(np.int32)
    elif pattern == "SEQ":
        return np.array([h % table_rows for h in range(n)], dtype=np.int32)
    elif pattern == "SAME":
        idx = np.zeros(n, dtype=np.int32)
        for pos in range(block):
            row = int(rng.choice(table_rows))
            idx[pos * H : (pos + 1) * H] = row
        return idx
    elif pattern == "STRIDE":
        stride = max(1, table_rows // H)
        return np.array([(h * stride) % table_rows for h in range(n)], dtype=np.int32)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def table_row(r, d):
    return np.array([((r + c) % 8 + 1) * 0.0625 for c in range(d)], dtype=np.float64)


def compute_golden_block(indices_block, block_size, emb_dim, hidden_block, gw_block):
    output = np.zeros((block_size, emb_dim), dtype=np.float64)
    for pos in range(block_size):
        idx = indices_block[pos * H : (pos + 1) * H]
        hid = hidden_block[pos * emb_dim : (pos + 1) * emb_dim]
        gw = gw_block[pos * emb_dim : (pos + 1) * emb_dim]
        gathered = np.zeros((H, emb_dim), dtype=np.float64)
        for h in range(H):
            gathered[h, :] = table_row(int(idx[h]), emb_dim)
        agg = gathered.mean(axis=0)
        dot = np.sum(hid * gw) + GATE_BIAS
        gate = 1.0 / (1.0 + np.exp(-dot))
        output[pos, :] = hid + gate * agg
    return output.flatten()


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    filter_name = sys.argv[2] if len(sys.argv) > 2 else None
    if filter_name and filter_name.startswith("ENGRAMSIMTTest."):
        filter_name = filter_name[len("ENGRAMSIMTTest.") :]

    configs = []
    if PERF_ANALYSIS:
        for emb in DIMS:
            for block in BLOCKS:
                for table_rows in TABLES[emb]:
                    tag = TABLE_TAGS[table_rows]
                    for pattern in PATTERNS:
                        configs.append((emb, block, table_rows, f"{tag}_{pattern}", pattern))
    else:
        for emb, block, table_rows, tag in DEFAULT_CONFIGS:
            configs.append((emb, block, table_rows, tag, "RAND"))

    count = 0
    for emb, block, table_rows, tag, pattern in configs:
        seed = emb * 100000 + block * 1000 + (table_rows % 10000) * 10 + PATTERN_IDX[pattern]
        rng = np.random.RandomState(seed & 0x7FFFFFFF)

        indices = gen_indices(pattern, block, table_rows, rng)

        hidden = np.full(block * emb, 0.25, dtype=np.float64)
        gw = np.zeros(block * emb, dtype=np.float64)
        for pos in range(block):
            for j in range(emb):
                gw[pos * emb + j] = 0.001953125 * (1 + j % 4)

        golden = compute_golden_block(indices, block, emb, hidden, gw)

        for variant in VARIANTS:
            test_name = f"{variant}_E{emb}_B{block}_{tag}"

            if filter_name and filter_name not in test_name:
                continue

            case_dir = os.path.join(base_dir, f"ENGRAMSIMTTest.{test_name}")
            os.makedirs(case_dir, exist_ok=True)

            indices.tofile(os.path.join(case_dir, "indices.bin"))
            hidden.astype(np.float32).tofile(os.path.join(case_dir, "hidden.bin"))
            gw.astype(np.float32).tofile(os.path.join(case_dir, "gate_weight.bin"))
            golden.astype(np.float32).tofile(os.path.join(case_dir, "golden.bin"))

            count += 1
            print(f"  [{count}] {test_name:50s} tbl={table_rows:>8}x{emb:<5} B={block}")

    print(f"\nGenerated {count} test cases.")


if __name__ == "__main__":
    main()
