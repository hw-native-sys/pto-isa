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

"""Remove regenerable memory dumps while retaining instruction and timing evidence.

The default is a dry run. A manifest records inode, size and modification time;
changed files and files open by a visible process are skipped when applying it.
"""

import argparse
import json
import os
from pathlib import Path
import re
import stat


# Simulator rotation inserts a numeric suffix before .dump (for example,
# ub.wr_log.1.dump); these files contain the same regenerable memory payload.
MEMORY_DUMP = re.compile(
    r"^core\d+\.(?:.*\.)?(ub|l1|l0[abc](?:mx)?|bt|fbv2?|pb|pt|smask|spidx|sb|ssb|vreg|reg\d+)"
    r"(?:\.(rd_log|wr_log))?(?:\.\d+)?\.dump$"
)


def snapshot(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not MEMORY_DUMP.fullmatch(path.name):
        return None
    return {
        "path": str(path.absolute()),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ino": info.st_ino,
        "dev": info.st_dev,
    }


def open_files():
    opened = set()
    inaccessible = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            descriptors = list((process / "fd").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError:
            inaccessible.append(int(process.name))
            continue
        for descriptor in descriptors:
            try:
                info = descriptor.stat()
                opened.add((info.st_dev, info.st_ino))
            except OSError:
                continue
    return opened, inaccessible


def cleanup(records, apply=False):
    opened, inaccessible = open_files()
    summary = {
        "apply": apply,
        "candidates": len(records),
        "eligible_files": 0,
        "eligible_bytes": 0,
        "deleted_files": 0,
        "deleted_bytes": 0,
        "skipped_changed": 0,
        "skipped_open": 0,
        "skipped_missing": 0,
        "inaccessible_processes": inaccessible,
    }
    for record in records:
        path = Path(record["path"])
        try:
            current = snapshot(path)
        except FileNotFoundError:
            summary["skipped_missing"] += 1
            continue
        if current != record:
            summary["skipped_changed"] += 1
            continue
        if (record["dev"], record["ino"]) in opened:
            summary["skipped_open"] += 1
            continue
        summary["eligible_files"] += 1
        summary["eligible_bytes"] += record["size"]
        if apply:
            path.unlink()
            summary["deleted_files"] += 1
            summary["deleted_bytes"] += record["size"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="Completed simulator run directory")
    source.add_argument("--manifest", type=Path, help="Saved candidate manifest")
    parser.add_argument("--apply", action="store_true", help="Delete unchanged eligible memory dump files")
    parser.add_argument("--save-manifest", type=Path)
    parser.add_argument("--summary", type=Path, help="Save full JSON details and print compact totals to the console")
    args = parser.parse_args()
    if args.manifest:
        records = json.loads(args.manifest.read_text())["files"]
    else:
        root = args.root.resolve(strict=True)
        if not root.is_dir() or root == Path("/"):
            parser.error("--root must be a simulator run directory")
        records = []
        for directory, subdirectories, files in os.walk(root, followlinks=False):
            subdirectories[:] = [name for name in subdirectories if name not in (".git", "fixture", "golden")]
            for name in files:
                if MEMORY_DUMP.fullmatch(name):
                    record = snapshot(Path(directory) / name)
                    if record:
                        records.append(record)
    if args.save_manifest:
        args.save_manifest.write_text(json.dumps({"regex": MEMORY_DUMP.pattern, "files": records}, indent=2) + "\n")
    summary = cleanup(records, args.apply)
    result = json.dumps(summary, indent=2) + "\n"
    if args.summary:
        args.summary.write_text(result)
        action = "deleted" if args.apply else "eligible"
        skipped = sum(summary[key] for key in ("skipped_changed", "skipped_open", "skipped_missing"))
        print(
            f"Memory dump cleanup: {summary[action + '_files']} files {action} "
            f"({summary[action + '_bytes'] / 1024**2:.1f} MiB); "
            f"{skipped} skipped; {len(summary['inaccessible_processes'])} inaccessible processes. "
            f"Details: {args.summary}"
        )
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
