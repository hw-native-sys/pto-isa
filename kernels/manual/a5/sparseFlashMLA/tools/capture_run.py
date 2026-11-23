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

"""Capture simulator run metadata using selected environment variables."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    info = path.stat()
    return {
        "sha256": digest.hexdigest(),
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "resolved_path": str(path.resolve()),
    }


def command_output(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {"command": command, "exit_status": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def capture(source, build, fixture, simulator, command, mode, soc):
    sources = {}
    pto_include = source.parents[3] / "include" / "pto"
    source_scope = {
        "kernel_root": str(source),
        "kernel_root_recursive": False,
        "recursive_directories": [str(source / name) for name in ("custom", "golden", "tools")] + [str(pto_include)],
        "file_types": [".h", ".hpp", ".cpp", ".py", ".sh", "CMakeLists.txt"],
    }
    directories = [
        (source, False),
        (source / "custom", True),
        (source / "golden", True),
        (source / "tools", True),
        (pto_include, True),
    ]
    for directory, recursive in directories:
        paths = directory.rglob("*") if recursive else directory.iterdir()
        for path in sorted(paths):
            if path.is_file() and (
                path.suffix in (".h", ".hpp", ".cpp", ".py", ".sh") or path.name == "CMakeLists.txt"
            ):
                if not any(part in ("build", "__pycache__", ".git") for part in path.relative_to(directory).parts):
                    sources[str(path)] = file_record(path)
    fixtures = {
        str(path): file_record(path)
        for path in sorted(fixture.iterdir())
        if path.is_file() and path.name not in ("output.bin", "output_lse.bin")
    }
    binaries = {}
    artifacts = [build / "CMakeCache.txt"]
    artifacts.extend(sorted((build / "lib").glob("*.so")))
    if command:
        artifacts.append(Path(command[0]))
    for path in artifacts:
        if path.is_file():
            binaries[str(path)] = file_record(path)
    compiler = None
    build_options = {}
    cache = build / "CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_CXX_COMPILER:"):
                compiler = line.split("=", 1)[1]
            if (
                line.startswith(("FLASH_MLA_", "RUN_MODE:", "SOC_VERSION:", "CMAKE_BUILD_TYPE:", "CMAKE_CXX_FLAGS"))
                and "=" in line
            ):
                key, value = line.split("=", 1)
                build_options[key.split(":", 1)[0]] = value
    runtime = {}
    toolkit = Path(os.environ["ASCEND_HOME_PATH"])
    directories = [toolkit / "lib64"]
    if mode == "sim":
        directories.insert(0, simulator)
    for directory in directories:
        for path in sorted(directory.iterdir()):
            if path.is_file() and (
                (directory == simulator and (".so" in path.name or path.suffix == ".json"))
                or path.name in ("libascendcl.so", "libruntime.so", "libruntime_camodel.so")
            ):
                runtime[str(path)] = file_record(path)
    result = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "run_mode": mode,
        "soc_version": soc,
        "toolkit_path": str(toolkit),
        "source_directory": str(source),
        "build_directory": str(build),
        "fixture_directory": str(fixture),
        "simulator_directory": str(simulator) if mode == "sim" else None,
        "sources": sources,
        "source_scope": source_scope,
        "build_options": build_options,
        "binaries_and_build_cache": binaries,
        "fixture_files": fixtures,
        "runtime_and_simulator": runtime,
        "git_revision": command_output(["git", "rev-parse", "HEAD"], source),
    }
    if compiler:
        result["compiler_version"] = command_output([compiler, "--version"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--golden-dir", required=True, type=Path)
    parser.add_argument("--sim-lib", required=True, type=Path)
    parser.add_argument("--run-mode", choices=("sim", "npu"), required=True)
    parser.add_argument("--soc-version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    manifest = capture(
        args.source_dir.resolve(),
        args.build_dir.resolve(),
        args.golden_dir.resolve(),
        args.sim_lib.resolve(),
        command,
        args.run_mode,
        args.soc_version,
    )
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
