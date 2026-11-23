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

"""Prepare cached private simulator libraries with a bounded physical core count."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def prepare(toolkit, soc, cores, cache):
    source = toolkit / "tools" / "simulator" / soc / "lib"
    if not source.is_dir():
        source = toolkit / "simulator" / soc / "lib"
    config = json.loads((source / "config.json").read_text())
    stars = json.loads((source / "config_stars.json").read_text())
    maximum = int(stars["model_top"]["num_aic"])
    if not 1 <= cores <= maximum:
        raise ValueError(f"--cores must be between 1 and {maximum}")
    libraries = sorted(path for path in source.iterdir() if path.is_file() and ".so" in path.name)
    digest = hashlib.sha256()
    digest.update(str(source.resolve()).encode())
    for path in [source / "config.json", source / "config_stars.json", *libraries]:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    identity = digest.hexdigest()
    target = cache / soc / identity[:16] / f"cores{cores}"
    if target.is_dir():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    config["LOG"]["core_enable_mask"] = [hex((1 << cores) - 1)]
    config["LOG"]["rotating_file_size"] = 1024 * 1024 * 1024
    stars["model_top"]["num_aic"] = cores
    stars["model_top"]["num_aiv"] = 2 * cores
    with tempfile.TemporaryDirectory(prefix="prepare.", dir=target.parent) as temporary:
        stage = Path(temporary) / "lib"
        stage.mkdir()
        for path in libraries:
            shutil.copy2(path, stage / path.name)
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (stage / "config_stars.json").write_text(json.dumps(stars, indent=2) + "\n")
        (stage / "private_simulator.json").write_text(
            json.dumps({"source": str(source), "source_sha256": identity, "physical_aic": cores}, indent=2) + "\n"
        )
        try:
            stage.rename(target)
        except FileExistsError:
            pass
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, default=os.environ.get("ASCEND_HOME_PATH"))
    parser.add_argument("--soc-version", default="Ascend950PR_9599")
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build" / "simulator",
        help="Private simulator cache directory (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.toolkit is None:
        parser.error("Set ASCEND_HOME_PATH or provide --toolkit")
    try:
        target = prepare(args.toolkit.resolve(), args.soc_version, args.cores, args.cache_dir.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(target)


if __name__ == "__main__":
    main()
