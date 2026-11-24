#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reproduce the 544-entry one-level-arch portfolio on CPU and native Linx.

Use the SuperNPUBench f6359b20 source tree (d34cea27 has identical benchmark
sources), Linx-TileOP-API 74d7326c and the matching Linx Clang/gfrun toolchain.
Each result records commands, input hashes, logs and numerical check results.
The CPU build adapter emits its exact source patch beside its isolated copy.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("linx_supernpubench")
sys.path.insert(0, str(FIXTURES))
from host_adapter import prepare as prepare_host  # noqa: E402


def execute(argv, cwd, log, timeout):
    start = time.monotonic()
    with log.open("wb") as stream:
        proc = subprocess.Popen(
            list(map(str, argv)), cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        timed = False
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed = True
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
    return dict(
        argv=list(map(str, argv)),
        cwd=str(cwd),
        log=str(log),
        returncode=code,
        timeout=timed,
        seconds=round(time.monotonic() - start, 3),
    )


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def provenance(args, bench):
    def revision(path):
        if not path:
            return None
        proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None

    def digest(root, suffixes):
        result = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.suffix in suffixes or path.name == "Makefile"):
                result.update(str(path.relative_to(root)).encode() + b"\0")
                result.update(path.read_bytes())
        return result.hexdigest()

    return dict(
        benchmark_revision=revision(args.benchmark),
        benchmark_source_sha256=digest(bench, {".cpp", ".hpp", ".h", ".c", ".py", ".mk", ".s"}),
        backend_revision=revision(ROOT),
        backend_headers_sha256=digest(ROOT / "include/pto", {".hpp", ".h"}),
        adapter_sha256=digest(FIXTURES, {".cpp", ".hpp", ".h", ".py", ".s", ".json"}),
        upstream_revision=revision(args.upstream),
        model_revision=revision(args.model),
        model_binary_sha256=hashlib.sha256((args.model / "bin/gfrun").read_bytes()).hexdigest() if args.model else None,
        compiler=args.compiler,
        toolchain=str(args.toolchain) if args.toolchain else None,
        sanitized=args.sanitize,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="SuperNPUBench checkout")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--musl-libs", type=Path)
    parser.add_argument("--routes", default="cpu,upstream")
    parser.add_argument("--ids", default="")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    for name in ("benchmark", "upstream", "toolchain", "model", "musl_libs"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    routes = args.routes.split(",")
    if any(r not in ("cpu", "upstream") for r in routes):
        parser.error("routes must be cpu and/or upstream")
    if "upstream" in routes and not all((args.upstream, args.toolchain, args.model, args.musl_libs)):
        parser.error("upstream requires --upstream, --toolchain, --model and --musl-libs")
    if args.jobs < 1 or args.timeout < 1:
        parser.error("jobs and timeout must be positive")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    bench = args.benchmark.resolve() / "benchmark/one-level-arch"
    source_info = provenance(args, bench)
    (output / "provenance.json").write_text(json.dumps(source_info, indent=2) + "\n")
    cases = json.loads((FIXTURES / "cases.json").read_text())
    if args.ids:
        ids = set(args.ids.split(","))
        if ids - {c["id"] for c in cases}:
            parser.error("unknown case ID")
        cases = [c for c in cases if c["id"] in ids]
    os.environ["LINX_HOST_CXX"] = shutil.which(args.compiler) or args.compiler
    host = output / "host/SuperNPUBench/benchmark/one-level-arch"
    if "cpu" in routes:
        prepare_host(bench, host)
        guard = bench / "test/solution/tileop-test/common/guard_io.c"
        record = execute(
            [args.compiler, "-x", "c", "-c", guard, "-o", output / "guard_io.o"], output, output / "guard-io.log", 60
        )
        if record["returncode"]:
            raise RuntimeError(f"guard_io compile failed: {record['log']}")
    if "upstream" in routes:
        record = execute(
            [
                args.toolchain / "bin/clang++",
                "-mlxbc",
                "-c",
                "-O2",
                FIXTURES / "group_worker.cpp",
                "-o",
                output / "group_worker.o",
            ],
            output,
            output / "group-worker.log",
            60,
        )
        if record["returncode"]:
            raise RuntimeError(f"group worker compile failed: {record['log']}")
    # Generate each input/golden set once, then copy it identically to both routes.
    import numpy as np

    base_oracle = module(bench / "test/kernel/res_check_all.py", "bench_oracle")
    matmul = module(bench / "test/kernel/matmul/src/gfrun_matmul.py", "matmul_oracle")
    if matmul._HAS_TORCH:
        matmul.torch.set_num_threads(1)
    for case in cases:
        if not case.get("numeric"):
            continue
        folder = output / "inputs" / case["id"]
        folder.mkdir(parents=True, exist_ok=True)
        n = case["numeric"]
        if "checker" in n:
            record = execute(
                [sys.executable, bench / n["checker"], "gen", n["name"], folder], output, folder / "prepare.log", 60
            )
            if record["returncode"]:
                raise RuntimeError(f"{case['id']} input generation failed")
        elif case["id"].startswith("N"):
            oracle_name = "matmul_reuseB" if case["id"] == "N0009" else case["name"]
            ref = next(c for c in base_oracle.CASES if c.name == oracle_name)
            np.save(folder / "golden.npy", np.asarray(ref.prepare(folder), dtype=ref.output_dtype).reshape(-1))
        else:
            matmul.gen_input_and_golden(case["name"], str(folder), SimpleNamespace(seed=n["seed"]))
            np.save(folder / "golden.npy", np.fromfile(folder / "golden.bin", dtype="f4"))
        (folder / n.get("output_name", "out.bin")).unlink(missing_ok=True)

    def run(case, route):
        folder = output / "results" / case["id"] / route
        folder.mkdir(parents=True, exist_ok=True)
        numeric = case.get("numeric")
        if numeric:
            shutil.copytree(output / "inputs" / case["id"], folder / "data", dirs_exist_ok=True)
            (folder / "data" / numeric.get("output_name", "out.bin")).unlink(missing_ok=True)
        settings = [a for a in case["args"] if not a.startswith("res_check=")]
        argv = ["make", "-B", "-j1", *settings, f"OBJ_ROOT={folder}/output", "all"]
        if route == "cpu":
            flags = (
                f"-I{FIXTURES}/bridge -I{ROOT}/include -ffp-contract=off -D__CPU_SIM -DPTO_CPU_SIM_TARGET_LINX -pthread"
            )
            opt = "-O1 -g1 -fsanitize=address,undefined -fno-omit-frame-pointer" if args.sanitize else "-O2"
            argv += [
                "PLAT=cpu",
                f"CXX={FIXTURES}/cxx.py",
                f"CC={args.compiler}",
                f"AS={args.compiler}",
                f"LINK={args.compiler}",
                "CXX_VER=-std=c++23",
                f"CC_O=-c {opt}",
                f"CC_OPTS={flags}",
                f"CC_LINK=-I{ROOT}/include -D__CPU_SIM -DPTO_CPU_SIM_TARGET_LINX -std=c++23 {opt} -pthread -ffp-contract=off -DPES={case['pes']} {FIXTURES}/launcher.cpp",
                "EXTRA_OBJ_FILES=",
            ]
        else:
            flags = f"-I{args.upstream}/include -ffp-contract=off -isystem {args.toolchain}/lib/clang/15.0.4/include/tileop-api"
            link = f"-static -nostdlib++ -L{args.musl_libs}"
            if not numeric:
                link += f" -nostartfiles {FIXTURES}/start.s"
            argv += [
                "PLAT=linx",
                f"COMPILER_DIR={args.toolchain}/bin",
                f"CC_OPTS={flags}",
                f"CC_LINK={link}",
                f"EXTRA_OBJ_FILES={output}/group_worker.o" if numeric else "EXTRA_OBJ_FILES=",
            ]
        argv += ["res_check=on", f"CHK_DIR={folder}/data"] if numeric else ["res_check=off"]
        if case["suite"].startswith("solution/tileop-test/"):
            defines = "-D__linx -DENABLE_TENSOR_INSTR" if route == "upstream" else "-D__cpu_sim__"
            if numeric:
                defines += f' -DRES_CHECK -DCHK_DIR=\\"{folder}/data\\"'
            argv += [f"DEFINES={defines}"]
            if route == "cpu":
                argv += [f"GUARD_IO_OBJ={output}/guard_io.o", f"--old-file={output}/guard_io.o"]
        record = dict(case=case["id"], route=route, pes=case["pes"], numeric=bool(numeric))
        record["input_hashes"] = (
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((folder / "data").glob("*.bin"))}
            if numeric
            else {}
        )
        record["build"] = execute(
            argv, (host if route == "cpu" else bench) / "test" / case["suite"], folder / "build.log", args.timeout
        )
        log = (folder / "build.log").read_text(errors="replace")
        if record["build"]["returncode"]:
            status = (
                "BUILD_TIMEOUT"
                if record["build"]["timeout"]
                else "COMPILER_CRASH"
                if any(s in log for s in ("PLEASE submit a bug report", "Stack dump:", "internal compiler error"))
                else "BUILD_FAIL"
            )
        else:
            (elf,) = (folder / "output").rglob("*.elf")
            record["elf"] = str(elf)
            argv = (
                [elf]
                if route == "cpu"
                else [args.model / "bin/gfrun", "-f", elf, "-s", f"softcore.multiThreadNum={case['pes']}"]
            )
            record["run"] = execute(argv, folder if route == "cpu" else args.model, folder / "run.log", args.timeout)
            log = (folder / "run.log").read_text(errors="replace")
            bad = re.search(r"\b(ERROR|FATAL|ASSERT|CPU_EXCEPTION)\b", log)
            ended = route == "cpu" or "Reach the End of Benchmark! R2 = 0" in log
            status = (
                "RUN_TIMEOUT"
                if record["run"]["timeout"]
                else "RUN_FAIL"
                if record["run"]["returncode"] or bad or not ended
                else "RUN_OK_UNCHECKED"
            )
            if numeric and status == "RUN_OK_UNCHECKED":
                if "checker" in numeric:
                    record["check"] = execute(
                        [sys.executable, bench / numeric["checker"], "check", numeric["name"], folder / "data"],
                        folder,
                        folder / "check.log",
                        60,
                    )
                    status = "NUMERIC_PASS" if record["check"]["returncode"] == 0 else "NUMERIC_FAIL"
                elif not (folder / "data" / numeric["output_name"]).exists():
                    status = "OUTPUT_MISSING"
                else:
                    actual = np.fromfile(folder / "data" / numeric["output_name"], dtype=numeric["dtype"])
                    golden = np.load(folder / "data/golden.npy")
                    ok = actual.shape == golden.shape and np.allclose(
                        actual, golden, atol=numeric["atol"], rtol=numeric["rtol"]
                    )
                    status = "NUMERIC_PASS" if ok else "NUMERIC_FAIL"
                    record["compare"] = dict(
                        actual_elements=actual.size,
                        golden_elements=golden.size,
                        atol=numeric["atol"],
                        rtol=numeric["rtol"],
                        passed=bool(ok),
                    )
                    if actual.shape == golden.shape:
                        record["compare"]["max_abs"] = float(np.max(np.abs(actual.astype("f8") - golden.astype("f8"))))
        record["status"] = status
        (folder / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(case["id"], route, status, flush=True)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, case, route) for case in cases for route in routes]
        for future in as_completed(futures):
            records.append(future.result())
    counts = {r: dict(Counter(x["status"] for x in records if x["route"] == r)) for r in routes}
    summary = dict(cases=cases, records=records, counts=counts, provenance=source_info)
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    # Portfolio includes documented upstream failures. The command fails only
    # when a native pass lacks a matching CPU pass, or execution is incomplete.
    good = {"RUN_OK_UNCHECKED", "NUMERIC_PASS"}
    by_key = {(x["case"], x["route"]): x["status"] for x in records}
    missing = [
        c["id"]
        for c in cases
        if by_key.get((c["id"], "upstream")) in good and "cpu" in routes and by_key.get((c["id"], "cpu")) not in good
    ]
    if missing:
        print("Native passes missing on CPU:", ",".join(missing))
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
