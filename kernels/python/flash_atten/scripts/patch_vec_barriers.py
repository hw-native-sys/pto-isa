#!/usr/bin/env python3
"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License.

Patch selected generated-C++ PIPE_V barriers for flash-attention experiments.
"""

import argparse
import re
import sys
from pathlib import Path

PATTERN_ALIASES = {
    "gu": "gu",
    "trowexpandmul-tadd": "gu",
    "softmax-exp-sum": "softmax-exp-sum",
    "texp-trowsum": "softmax-exp-sum",
    "softmax-sum-add": "softmax-sum-add",
    "trowsum-tadd": "softmax-sum-add",
}
DISABLED_PATTERNS = {"none", "off", "disable", "disabled"}
CALL_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\s*(?:<[^;]*)?\((.*)\);\s*$")
TILE_VAR_RE = re.compile(r"\bv\d+\b")


def _parse_patterns(raw_patterns):
    raw_pattern_set = {
        part.strip() for part in raw_patterns.split(",") if part.strip() and part.strip() not in DISABLED_PATTERNS
    }
    unknown_patterns = sorted(raw_pattern_set - set(PATTERN_ALIASES))
    if unknown_patterns:
        supported = ", ".join(sorted(PATTERN_ALIASES))
        raise SystemExit(
            "unknown --remove-vec-barrier-patterns entries: {}; supported: {}".format(unknown_patterns, supported)
        )
    return {PATTERN_ALIASES[part] for part in raw_pattern_set}


def _parse_lines(raw_lines):
    try:
        return {int(part.strip()) for part in raw_lines.split(",") if part.strip()}
    except ValueError as exc:
        raise SystemExit("invalid --remove-vec-barriers list: {!r}: {}".format(raw_lines, exc)) from exc


def _op_name(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("//"):
        return ""
    head = stripped.split("(", 1)[0]
    return head.split("<", 1)[0].strip()


def _call_tile_vars(line):
    match = CALL_RE.match(line)
    if not match:
        return []
    return TILE_VAR_RE.findall(match.group(2))


def _has_direct_tile_dependency(prev_line, next_line):
    prev_vars = _call_tile_vars(prev_line)
    next_vars = _call_tile_vars(next_line)
    if not prev_vars or not next_vars:
        return False

    prev_writes = {prev_vars[0]}
    prev_reads = set(prev_vars[1:])
    next_writes = {next_vars[0]}
    next_reads = set(next_vars[1:])

    return bool((prev_writes & next_reads) or (prev_writes & next_writes) or (prev_reads & next_writes))


def _collect_pattern_lines(lines, patterns):
    pattern_lines = {}
    skipped_pattern_counts = {}

    def add_pattern_line(lineno, reason):
        pattern_lines.setdefault(lineno, set()).add(reason)

    def skip_pattern(reason):
        skipped_pattern_counts[reason] = skipped_pattern_counts.get(reason, 0) + 1

    if "gu" in patterns:
        for idx, line in enumerate(lines):
            if _op_name(line) != "TROWEXPANDMUL":
                continue
            j = idx + 1
            candidate = None
            while j < len(lines):
                op = _op_name(lines[j])
                if op == "pipe_barrier" and lines[j].strip() == "pipe_barrier(PIPE_V);" and candidate is None:
                    candidate = j + 1
                    j += 1
                    continue
                if op == "wait_flag" and "PIPE_MTE2, PIPE_V" in lines[j]:
                    j += 1
                    continue
                if op == "TADD" and candidate is not None:
                    add_pattern_line(candidate, "gu")
                break

    if "softmax-exp-sum" in patterns:
        for idx, line in enumerate(lines):
            if _op_name(line) != "TEXP":
                continue
            barrier_lineno = idx + 2
            next_idx = idx + 1
            if next_idx >= len(lines) or lines[next_idx].strip() != "pipe_barrier(PIPE_V);":
                continue
            after_idx = next_idx + 1
            if after_idx < len(lines) and _op_name(lines[after_idx]) == "TROWSUM":
                if _has_direct_tile_dependency(line, lines[after_idx]):
                    skip_pattern("softmax-exp-sum:direct-tile-dependency")
                else:
                    add_pattern_line(barrier_lineno, "softmax-exp-sum")

    if "softmax-sum-add" in patterns:
        for idx, line in enumerate(lines):
            if _op_name(line) != "TROWSUM":
                continue
            barrier_lineno = idx + 2
            next_idx = idx + 1
            if next_idx >= len(lines) or lines[next_idx].strip() != "pipe_barrier(PIPE_V);":
                continue
            after_idx = next_idx + 1
            if after_idx < len(lines) and _op_name(lines[after_idx]) == "TADD":
                if _has_direct_tile_dependency(line, lines[after_idx]):
                    skip_pattern("softmax-sum-add:direct-tile-dependency")
                else:
                    add_pattern_line(barrier_lineno, "softmax-sum-add")

    return pattern_lines, skipped_pattern_counts


def patch_file(src, dst, raw_lines="", raw_patterns=""):
    patterns = _parse_patterns(raw_patterns)
    experimental_patterns = sorted(patterns - {"gu"})
    if experimental_patterns:
        print(
            "warning: experimental PIPE_V barrier pattern(s) may affect benchmark timing or performance: "
            + ",".join(experimental_patterns),
            file=sys.stderr,
        )

    remove_lines = _parse_lines(raw_lines)
    lines = src.read_text().splitlines()
    pattern_lines, skipped_pattern_counts = _collect_pattern_lines(lines, patterns)

    patched = []
    seen = set()
    bad_lines = []
    for lineno, line in enumerate(lines, start=1):
        by_line = lineno in remove_lines
        by_pattern = lineno in pattern_lines
        if by_line or by_pattern:
            if by_line:
                seen.add(lineno)
            if line.strip() != "pipe_barrier(PIPE_V);":
                bad_lines.append((lineno, line.strip()))
                patched.append(line)
            else:
                indent = line[: len(line) - len(line.lstrip())]
                if by_line:
                    reason = "--remove-vec-barriers"
                else:
                    reason = "--remove-vec-barrier-patterns=" + ",".join(sorted(pattern_lines[lineno]))
                patched.append("{}/* removed PIPE_V barrier via {} */".format(indent, reason))
        else:
            patched.append(line)

    missing = sorted(remove_lines - seen)
    if missing or bad_lines:
        if missing:
            print("line numbers outside generated file: {}".format(missing), file=sys.stderr)
        for lineno, text in bad_lines:
            print("line {} is not a PIPE_V barrier: {!r}".format(lineno, text), file=sys.stderr)
        raise SystemExit(2)

    dst.write_text("\n".join(patched) + "\n")
    pattern_total = len(pattern_lines)
    removed_total = len(remove_lines | set(pattern_lines))
    print(
        "Patched generated C++ -> {} (removed {} PIPE_V barriers; lines={}, patterns={})".format(
            dst, removed_total, len(remove_lines), pattern_total
        )
    )
    if skipped_pattern_counts:
        skipped = ", ".join("{}={}".format(reason, count) for reason, count in sorted(skipped_pattern_counts.items()))
        print("Skipped PIPE_V barrier pattern candidates: {}".format(skipped))


def _parse_args():
    parser = argparse.ArgumentParser(description="Patch selected generated-C++ PIPE_V barriers.")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--remove-vec-barriers", default="")
    parser.add_argument("--remove-vec-barrier-patterns", default="")
    return parser.parse_args()


def main():
    args = _parse_args()
    patch_file(args.src, args.dst, args.remove_vec_barriers, args.remove_vec_barrier_patterns)


if __name__ == "__main__":
    main()
