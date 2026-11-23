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

"""Summarize FlashMLA correctness, elapsed ticks, local VF work and matmul queues.

Tick values use their simulator domains; local VF work can overlap in time.
QK/PV classification uses the kernel's L0C layout contract.
Executable exit status and model completion are recorded separately.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


FOOTER = re.compile(r"core:\s*(\d+),\s*subcore:\s*(\d+),\s*start:\s*(\d+),\s*tick:\s*(\d+)")
METRIC = re.compile(r"([\w.()]+)\s*\|\s*(\d+)\s*$", re.MULTILINE)
TICK = re.compile(r"\bkern(?:al|el) total ticks\s*:\s*(\d+)")
SYSTEM = re.compile(r"\bsystem total ticks\s*:\s*(\d+)")
SCALAR_KINDS = ("ldst", "exe", "variable", "float", "jump", "pipe", "others")
CHECK_NAMES = ("mathematical output", "mathematical LSE", "device-model output", "device-model LSE")
VF = re.compile(
    r"\[(\d+)\].*\(PC:\s*(0x[\da-fA-F]+)\).*\(ID:\s*(\d+)\)\s+VF\s*,"
    r"\s*vf_execute_time:\s*(\d+),\s*vf_real_execute_time:\s*(\d+)"
)
MATMUL = re.compile(
    r"\[(\d+)\].*\(PC:\s*(0x[\da-fA-F]+)\).*\(ID:\s*(\d+)\)\s+"
    r"(MMAD(?:_MX)?)\s+dtype:(\S+)\s+XD:(0x[\da-fA-F]+)"
)
QUEUE = re.compile(
    r"\[info\]\s+(\d+)\s+\[(Push Instr|pop Instr|RETIRE INSTR)\]\s+"
    r"(?:instr\.)?name=(MMAD(?:_MX)?)\s+(?:instr\.)?pc=(0x[\da-fA-F]+)\s+(?:instr\.)?id=(\d+)"
)


def read(path):
    return path.read_text(errors="replace") if path.is_file() else ""


def core_name(core, subcore):
    return f"core{core}.cubecore0" if subcore == 0 else f"core{core}.veccore{subcore - 1}"


def summary_sections(path):
    text = read(path)
    begin = 0
    for footer in FOOTER.finditer(text):
        section = text[begin : footer.start()]
        begin = footer.end()
        core, subcore, start, end = map(int, footer.groups())
        metrics = {match[1]: int(match[2]) for match in METRIC.finditer(section)}
        kernel = TICK.findall(section)
        system = SYSTEM.findall(section)
        yield (
            core_name(core, subcore),
            {
                "kernel_elapsed_ticks": int(kernel[0]) if len(kernel) == 1 else None,
                "system_ticks": int(system[0]) if len(system) == 1 else None,
                "start": start,
                "footer_tick": end,
                "metrics": metrics,
                "source": str(path),
            },
        )


def parse_timing(directory):
    cores = defaultdict(list)
    for path in sorted(directory.glob("core*_*summary_log")):
        if "_su_perf_summary_log" not in path.name:
            continue
        for name, section in summary_sections(path):
            scalar = {}
            for key, value in section.pop("metrics").items():
                if "scalar_" in key:
                    scalar[key.split("_su_", 1)[-1]] = value
            section["scalar"] = scalar
            counts = [scalar.get(f"issue_{kind}_instr_num") for kind in SCALAR_KINDS]
            section["scalar_instruction_count"] = sum(counts) if all(value is not None for value in counts) else None
            section["vector_busy_cycles"] = {}
            cores[name].append(section)
    for path in sorted(directory.glob("core[0-9]*_summary_log")):
        for name, section in summary_sections(path):
            matching = [item for item in cores[name] if item["start"] == section["start"]]
            for item in matching:
                item["vector_busy_cycles"] = {
                    key: value
                    for key, value in section["metrics"].items()
                    if key.startswith("rvec_") and "busy_cycle" in key
                }
    elapsed = [item["kernel_elapsed_ticks"] for items in cores.values() for item in items]
    known = [value for value in elapsed if value is not None]
    return dict(cores), max(known) if known and len(known) == len(elapsed) else None


def parse_correctness(directory, wrapper_exit):
    stdout = read(directory / "stdout.log")
    stderr = read(directory / "stderr.log")
    checks = {}
    for name in CHECK_NAMES:
        lines = re.findall(rf"^{re.escape(name)}:\s*(\S+)([^\n]*)", stdout, re.MULTILINE)
        records = []
        for status, detail in lines:
            fields = {}
            for key, value in re.findall(r"([a-z_]+)=([-+\d.eE]+)", detail):
                fields[key] = float(value) if any(letter in value.lower() for letter in (".", "e")) else int(value)
            records.append({"status": status, "values": fields})
        checks[name] = records
    recorded = read(directory / "exit_status.txt").strip()
    executable_exit = int(recorded) if re.fullmatch(r"-?\d+", recorded) else None
    caveats = []
    if executable_exit is None:
        caveats.append("Executable exit status is unknown; model shutdown and comparison checks are separate evidence.")
    if wrapper_exit is not None:
        caveats.append(
            f"Wrapper exit status {wrapper_exit} supplied explicitly; executable status is recorded separately."
        )
    if "syntax error" in stderr:
        caveats.append("Wrapper syntax error appears in stderr; preserve it separately from kernel results.")
    all_checks = all(
        len(records) == 1
        and records[0]["status"] in ("PASS", "APPROX_WITHIN_TOLERANCE")
        and records[0]["values"].get("nonfinite_mismatches", 0) == 0
        and records[0]["values"].get("mismatches", records[0]["values"].get("threshold_exceedances", 0)) == 0
        for records in checks.values()
    )
    return {
        "checks": checks,
        "all_four_checks_pass": all_checks,
        "implementation_pass": "Implementation correctness: PASS" in stdout,
        "model_stopped_successfully": "Model stopped successfully" in stdout,
        "block_end_records": len(re.findall(r"\[block_end\]", stdout)),
        "executable_exit_status": executable_exit,
        "wrapper_exit_status_override": wrapper_exit,
        "host_total_tick_values": [int(value) for value in re.findall(r"Total tick:\s*(\d+)", stdout)],
        "error_line_count": len(re.findall(r"^.*(?:\[error\]|error:|syntax error).*$", stdout + stderr, re.I | re.M)),
        "caveats": caveats,
    }


def parse_instructions(path):
    vf = defaultdict(lambda: {"count": 0, "execute_ticks_sum": 0, "real_ticks_sum": 0})
    matmuls = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            match = VF.search(line)
            if match:
                tick, pc, identity, execute, real = match.groups()
                group = vf[pc.lower()]
                group["count"] += 1
                group["execute_ticks_sum"] += int(execute)
                group["real_ticks_sum"] += int(real)
                group.setdefault("first_instruction_tick", int(tick))
                group["last_instruction_tick"] = int(tick)
            match = MATMUL.search(line)
            if match:
                tick, pc, identity, operation, dtype, destination = match.groups()
                if int(identity) in matmuls:
                    raise ValueError(
                        f"Repeated matmul ID {identity} in {path}; separate task invocations before joining queues"
                    )
                matmuls[int(identity)] = {
                    "id": int(identity),
                    "pc": pc.lower(),
                    "operation": operation,
                    "dtype": dtype,
                    "l0c_destination": int(destination, 16),
                    "instruction_tick": int(tick),
                }
    return dict(vf), matmuls


def parse_queue(path, matmuls, start):
    event_keys = {"Push Instr": "enqueue", "pop Instr": "accept", "RETIRE INSTR": "retire"}
    with path.open(errors="replace") as stream:
        for line in stream:
            match = QUEUE.search(line)
            if not match:
                continue
            tick, kind, operation, pc, identity = match.groups()
            event = matmuls.get(int(identity))
            if event and event["pc"] == pc.lower() and event["operation"] == operation:
                event[event_keys[kind]] = int(tick)
    for event in matmuls.values():
        event["relative_ticks"] = {
            key: event[key] - start if start is not None and key in event else None
            for key in ("enqueue", "accept", "retire")
        }
        event["enqueue_to_retire"] = (
            event["retire"] - event["enqueue"] if "retire" in event and "enqueue" in event else None
        )
    return list(matmuls.values())


def summarize(directory, tile_n=None, wrapper_exit=None):
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    stdout = read(directory / "stdout.log")
    if tile_n is None:
        selected = manifest.get("build_options", {}).get("FLASH_MLA_TILE_N")
        if selected is None:
            match = re.search(r"\btile_n=(\d+)", stdout)
            selected = match[1] if match else None
        tile_n = int(selected) if selected is not None else None
    timing, maximum = parse_timing(directory)
    vf = {}
    queues = {}
    for path in sorted(directory.glob("core*.instr_log.dump")):
        name = path.name.removesuffix(".instr_log.dump")
        groups, matmuls = parse_instructions(path)
        if groups:
            vf[name] = {
                "by_callsite_pc": groups,
                "real_ticks_sum": sum(group["real_ticks_sum"] for group in groups.values()),
            }
        queue = directory / f"{name}.ccu.cube_issque.dump"
        if not matmuls or not queue.is_file():
            continue
        sections = timing.get(name, [])
        start = sections[0]["start"] if len(sections) == 1 else None
        events = parse_queue(queue, matmuls, start)
        stages = {}
        if tile_n is not None:
            threshold = 2 * 64 * tile_n * 4
            for stage in ("QK", "PV"):
                matches = [item for item in events if (item["l0c_destination"] < threshold) == (stage == "QK")]
                ordered = sorted(matches, key=lambda item: item.get("enqueue", item["instruction_tick"]))
                stages[stage] = {
                    "count": len(matches),
                    "queue_complete_count": sum(
                        all(key in item for key in ("enqueue", "accept", "retire")) for item in matches
                    ),
                    "first": ordered[0] if ordered else None,
                    "last": ordered[-1] if ordered else None,
                }
        queues[name] = {"matmul_count": len(events), "stages": stages}
    return {
        "directory": str(directory),
        "tile_n": tile_n,
        "correctness_and_completion": parse_correctness(directory, wrapper_exit),
        "per_core": timing,
        "max_kernel_elapsed_ticks": maximum,
        "vector_functions": vf,
        "matmul_queues": queues,
        "mapping_contract": "M=64; QK L0C XD < 2*M*N*4 bytes; PV XD >= threshold; join by ID, PC and operation within core.",
        "limitations": [
            "VF entries identify call sites; local VF work can overlap, while kernel elapsed ticks measure duration.",
            "Queue push is enqueue, pop is acceptance, and RETIRE is completion; timings use raw simulator ticks.",
            "Matmul IDs must be unique within a core log; multiple task invocations require separate run directories.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--tile-n", type=int, choices=(64, 128, 256))
    parser.add_argument(
        "--wrapper-exit-status", type=int, help="Known wrapper status, kept distinct from executable exit"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "runs": [summarize(path, args.tile_n, args.wrapper_exit_status) for path in args.run_dirs],
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
