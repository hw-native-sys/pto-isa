#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Check PTO Virtual ISA manual completeness and consistency.

Checks:
- Required EN/ZH manual chapters and entry pages exist.
- Required chapter headings exist.
- MkDocs nav order includes the expected chapter sequence (EN/ZH).
- EN/ZH manual files are standalone (no cross-language manual links).
- Appendix D coverage includes all manifest instructions exactly once (EN/ZH).
- Header inventory and manifest inventory remain aligned.
- Appendix D generation is synchronized.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
MANUAL_DIR = REPO_ROOT / "docs" / "mkdocs" / "src" / "manual"
MKDOCS_YML = REPO_ROOT / "docs" / "mkdocs" / "mkdocs.yml"
MANIFEST_PATH = REPO_ROOT / "docs" / "isa" / "manifest.yaml"
HEADER_PATH = REPO_ROOT / "include" / "pto" / "common" / "pto_instr.hpp"
ENTRY_EN = REPO_ROOT / "docs" / "PTO-Virtual-ISA-Manual.md"
ENTRY_ZH = REPO_ROOT / "docs" / "PTO-Virtual-ISA-Manual_zh.md"
APP_D_EN = MANUAL_DIR / "appendix-d-instruction-family-matrix.md"
APP_D_ZH = MANUAL_DIR / "appendix-d-instruction-family-matrix_zh.md"

EXPECTED_MANUAL_EN = [
    "index.md",
    "01-overview.md",
    "02-machine-model.md",
    "03-state-and-types.md",
    "04-tiles-and-globaltensor.md",
    "05-synchronization.md",
    "06-instructions.md",
    "07-programming.md",
    "08-virtual-isa-and-ir.md",
    "09-bytecode-and-toolchain.md",
    "10-memory-ordering-and-consistency.md",
    "11-backend-profiles-and-conformance.md",
    "appendix-a-glossary.md",
    "appendix-b-instruction-contract-template.md",
    "appendix-c-diagnostics-taxonomy.md",
    "appendix-d-instruction-family-matrix.md",
]

EXPECTED_MANUAL_ZH = [
    "index_zh.md",
    "01-overview_zh.md",
    "02-machine-model_zh.md",
    "03-state-and-types_zh.md",
    "04-tiles-and-globaltensor_zh.md",
    "05-synchronization_zh.md",
    "06-instructions_zh.md",
    "07-programming_zh.md",
    "08-virtual-isa-and-ir_zh.md",
    "09-bytecode-and-toolchain_zh.md",
    "10-memory-ordering-and-consistency_zh.md",
    "11-backend-profiles-and-conformance_zh.md",
    "appendix-a-glossary_zh.md",
    "appendix-b-instruction-contract-template_zh.md",
    "appendix-c-diagnostics-taxonomy_zh.md",
    "appendix-d-instruction-family-matrix_zh.md",
]

EXPECTED_HEADINGS: Dict[str, List[str]] = {
    "index.md": ["# PTO Virtual Instruction Set Architecture Manual", "## 0.1 Scope", "## 0.4 Conformance language"],
    "index_zh.md": ["# PTO 虚拟指令集架构手册", "## 0.1 范围", "## 0.4 规范性术语"],
    "10-memory-ordering-and-consistency.md": [
        "# 10. Memory Ordering and Consistency",
        "## 10.1 Scope",
        "## 10.4 Ordering guarantees",
    ],
    "10-memory-ordering-and-consistency_zh.md": ["# 10. 内存顺序与一致性", "## 10.1 范围", "## 10.4 顺序保证"],
    "11-backend-profiles-and-conformance.md": [
        "# 11. Backend Profiles and Conformance",
        "## 11.1 Scope",
        "## 11.5 Conformance levels",
    ],
    "11-backend-profiles-and-conformance_zh.md": ["# 11. 后端画像与一致性", "## 11.1 范围", "## 11.5 一致性等级"],
}

EXPECTED_NAV_EN = [
    "manual/index.md",
    "manual/01-overview.md",
    "manual/02-machine-model.md",
    "manual/03-state-and-types.md",
    "manual/04-tiles-and-globaltensor.md",
    "manual/05-synchronization.md",
    "manual/06-instructions.md",
    "manual/07-programming.md",
    "manual/08-virtual-isa-and-ir.md",
    "manual/09-bytecode-and-toolchain.md",
    "manual/10-memory-ordering-and-consistency.md",
    "manual/11-backend-profiles-and-conformance.md",
    "manual/appendix-a-glossary.md",
    "manual/appendix-b-instruction-contract-template.md",
    "manual/appendix-c-diagnostics-taxonomy.md",
    "manual/appendix-d-instruction-family-matrix.md",
]

EXPECTED_NAV_ZH = [
    "manual/index_zh.md",
    "manual/01-overview_zh.md",
    "manual/02-machine-model_zh.md",
    "manual/03-state-and-types_zh.md",
    "manual/04-tiles-and-globaltensor_zh.md",
    "manual/05-synchronization_zh.md",
    "manual/06-instructions_zh.md",
    "manual/07-programming_zh.md",
    "manual/08-virtual-isa-and-ir_zh.md",
    "manual/09-bytecode-and-toolchain_zh.md",
    "manual/10-memory-ordering-and-consistency_zh.md",
    "manual/11-backend-profiles-and-conformance_zh.md",
    "manual/appendix-a-glossary_zh.md",
    "manual/appendix-b-instruction-contract-template_zh.md",
    "manual/appendix-c-diagnostics-taxonomy_zh.md",
    "manual/appendix-d-instruction-family-matrix_zh.md",
]


def load_manifest(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("instructions", [])
    if not isinstance(entries, list):
        raise ValueError("manifest 'instructions' must be a list")
    return entries


def parse_header_instr(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    names = re.findall(r"PTO_INST\s+(?:void|RecordEvent)\s+([A-Z][A-Z0-9_]+)\s*\(", text)
    out: List[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def check_required_files(errors: List[str]) -> None:
    for rel in EXPECTED_MANUAL_EN + EXPECTED_MANUAL_ZH:
        path = MANUAL_DIR / rel
        if not path.exists():
            errors.append(f"missing required manual file: {path}")
    for path in (ENTRY_EN, ENTRY_ZH):
        if not path.exists():
            errors.append(f"missing top-level manual entry: {path}")


def check_required_headings(errors: List[str]) -> None:
    for rel, headings in EXPECTED_HEADINGS.items():
        path = MANUAL_DIR / rel
        if not path.exists():
            continue
        text = read(path)
        for heading in headings:
            if heading not in text:
                errors.append(f"missing heading '{heading}' in {path}")


def extract_nav_manual_paths(yml_text: str, zh: bool) -> List[str]:
    lines = yml_text.splitlines()
    start_tokens = (
        ("- PTO Virtual ISA Manual (ZH):", "- PTO ISA Manual (ZH):")
        if zh
        else ("- PTO Virtual ISA Manual:", "- PTO ISA Manual:")
    )
    end_tokens = (
        "- PTO Virtual ISA Manual (ZH):",
        "- PTO ISA Manual (ZH):",
        "- Programming Model:",
        "- Programming Model (ZH):",
        "- ISA Reference:",
        "- Machine Model:",
        "- Examples:",
        "- Documentation:",
        "- Full Index:",
    )

    collecting = False
    out: List[str] = []
    for raw in lines:
        line = raw.rstrip()
        if not collecting:
            if line.strip() in start_tokens:
                collecting = True
            continue

        if any(line.strip().startswith(token) for token in end_tokens):
            break

        match = re.search(r"(manual/[A-Za-z0-9_.\-]+\.md)", line)
        if match:
            out.append(match.group(1))
    return out


def check_nav_order(errors: List[str]) -> None:
    text = read(MKDOCS_YML)
    nav_en = extract_nav_manual_paths(text, zh=False)
    nav_zh = extract_nav_manual_paths(text, zh=True)

    if nav_en != EXPECTED_NAV_EN:
        errors.append("manual nav order mismatch (EN):\n" + f"expected: {EXPECTED_NAV_EN}\n" + f"actual:   {nav_en}")
    if nav_zh != EXPECTED_NAV_ZH:
        errors.append("manual nav order mismatch (ZH):\n" + f"expected: {EXPECTED_NAV_ZH}\n" + f"actual:   {nav_zh}")


def check_standalone_language_policy(errors: List[str]) -> None:
    en_files = [MANUAL_DIR / rel for rel in EXPECTED_MANUAL_EN] + [ENTRY_EN]
    zh_files = [MANUAL_DIR / rel for rel in EXPECTED_MANUAL_ZH] + [ENTRY_ZH]

    for path in en_files:
        if not path.exists():
            continue
        text = read(path)
        if "_zh.md" in text:
            errors.append(f"cross-language reference found in EN manual file: {path}")

    for path in zh_files:
        if not path.exists():
            continue
        text = read(path)
        if re.search(r"manual/[A-Za-z0-9_.\-]+\.md", text) and not re.search(r"manual/[A-Za-z0-9_.\-]+_zh\.md", text):
            errors.append(f"potential EN manual reference found in ZH manual file: {path}")
        if "PTO-Virtual-ISA-Manual.md" in text:
            errors.append(f"cross-language top-level reference found in ZH manual file: {path}")


def check_matrix_coverage(errors: List[str]) -> None:
    entries = load_manifest(MANIFEST_PATH)
    manifest_instr = [str(item.get("instruction", "")).strip() for item in entries]
    manifest_set = {instr for instr in manifest_instr if instr}

    text_en = read(APP_D_EN) if APP_D_EN.exists() else ""
    text_zh = read(APP_D_ZH) if APP_D_ZH.exists() else ""

    hit_en = re.findall(r"`([A-Z][A-Z0-9_]+)`", text_en)
    hit_zh = re.findall(r"`([A-Z][A-Z0-9_]+)`", text_zh)

    def exactly_once(hits: List[str], label: str) -> None:
        counts: Dict[str, int] = {}
        for name in hits:
            counts[name] = counts.get(name, 0) + 1
        missing = sorted(manifest_set - set(counts))
        extra = sorted(set(counts) - manifest_set)
        dup = sorted(name for name, cnt in counts.items() if cnt != 1 and name in manifest_set)

        if missing:
            errors.append(f"appendix D {label} missing instructions: {', '.join(missing)}")
        if extra:
            errors.append(f"appendix D {label} has non-manifest instructions: {', '.join(extra)}")
        if dup:
            errors.append(f"appendix D {label} duplicated instructions: {', '.join(dup)}")

    exactly_once(hit_en, "EN")
    exactly_once(hit_zh, "ZH")


def check_header_manifest_alignment(errors: List[str]) -> None:
    entries = load_manifest(MANIFEST_PATH)
    manifest_set = {str(item.get("instruction", "")).strip() for item in entries}
    header_set = set(parse_header_instr(HEADER_PATH))

    missing = sorted(header_set - manifest_set)
    extra = sorted(manifest_set - header_set)

    if missing:
        errors.append("header instructions missing in manifest: " + ", ".join(missing))
    if extra:
        errors.append("manifest instructions missing in header: " + ", ".join(extra))


def run_matrix_sync_check(errors: List[str]) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "docs" / "tools" / "gen_virtual_manual_matrix.py"), "--check"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        errors.append("virtual manual matrix check failed:\n" + out.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PTO Virtual ISA manual consistency")
    parser.add_argument("--check", action="store_true", help="compatibility flag; checks are always performed")
    _ = parser.parse_args()

    errors: List[str] = []

    check_required_files(errors)
    check_required_headings(errors)
    check_nav_order(errors)
    check_standalone_language_policy(errors)
    check_matrix_coverage(errors)
    check_header_manifest_alignment(errors)
    run_matrix_sync_check(errors)

    if errors:
        print("Virtual manual consistency check failed:")
        for idx, err in enumerate(errors, start=1):
            print(f"  {idx}. {err}")
        return 1

    print("OK: virtual manual consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
