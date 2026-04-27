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

"""Check PTO ISA documentation consistency.

Checks:
- header instruction inventory vs manifest
- manifest vs docs/isa English pages
- Chinese pages exist and are standalone (no EN cross-links)
- English pages are standalone (no ZH cross-links)
- manifest vs per-instruction SVG availability
- generated index files are synchronized (`gen_isa_indexes.py --check`)
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
MANIFEST_PATH = REPO_ROOT / "docs" / "isa" / "manifest.yaml"
ISA_DIR = REPO_ROOT / "docs" / "isa"
SVG_DIR = REPO_ROOT / "docs" / "figures" / "isa"
PTO_HEADER = REPO_ROOT / "include" / "pto" / "common" / "pto_instr.hpp"
DOC_ONLY_MANIFEST_INSTRUCTIONS = {"SETTF32MODE", "SETTF32MODE"}


def load_manifest(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("instructions", [])
    if not isinstance(entries, list):
        raise ValueError("manifest 'instructions' must be a list")
    return entries


def header_instructions(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    names = re.findall(r"PTO_INST\s+(?:void|RecordEvent)\s+([A-Z][A-Z0-9_]+)\s*\(", text)
    out: List[str] = []
    seen: set[str] = set()
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def docs_instructions(isa_dir: Path) -> set[str]:
    return {
        p.stem
        for p in isa_dir.glob("*.md")
        if p.name not in {"README.md", "README_zh.md", "conventions.md", "conventions_zh.md"} and not p.stem.endswith("_zh")
    }


def check_en_page(instr: str, page: Path) -> List[str]:
    errors: List[str] = []
    text = page.read_text(encoding="utf-8", errors="ignore")
    svg_token = f"../figures/isa/{instr}.svg"
    if "_zh.md" in text:
        errors.append(f"unexpected zh reference in English page {page}")
    if svg_token not in text:
        errors.append(f"missing svg link in {page}: expected token '{svg_token}'")
    if "### IR Level 1 (SSA)" not in text and "### AS Level 1 (SSA)" not in text:
        errors.append(f"missing IR Level 1 syntax section in {page}")
    if "### IR Level 2 (DPS)" not in text and "### AS Level 2 (DPS)" not in text:
        errors.append(f"missing IR Level 2 syntax section in {page}")
    return errors


def check_zh_page(instr: str, page: Path) -> List[str]:
    errors: List[str] = []
    text = page.read_text(encoding="utf-8", errors="ignore")
    svg_token = f"../figures/isa/{instr}.svg"
    en_token = f"{instr}.md"
    if en_token in text:
        errors.append(f"unexpected en reference in Chinese page {page}: found token '{en_token}'")
    if svg_token not in text:
        errors.append(f"missing svg link in {page}: expected token '{svg_token}'")
    if not any(token in text for token in ("IR Level 1", "AS Level 1", "AS Level 1（SSA）", "IR Level 1（SSA）")):
        errors.append(f"missing IR Level 1 syntax section in {page}")
    if not any(token in text for token in ("IR Level 2", "AS Level 2", "AS Level 2（DPS）", "IR Level 2（DPS）")):
        errors.append(f"missing IR Level 2 syntax section in {page}")
    return errors


def run_index_check() -> List[str]:
    cmd = [sys.executable, str(REPO_ROOT / "docs" / "tools" / "gen_isa_indexes.py"), "--check"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode == 0:
        return []
    out = (proc.stdout or "") + (proc.stderr or "")
    return [f"index check failed:\n{out.strip()}"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ISA docs consistency")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    manifest_instrs = [str(e.get("instruction", "")).strip() for e in entries]

    errors: List[str] = []

    seen: set[str] = set()
    for instr in manifest_instrs:
        if not instr:
            errors.append("manifest has empty instruction name")
            continue
        if instr in seen:
            errors.append(f"duplicate instruction in manifest: {instr}")
        seen.add(instr)

    header_instrs = header_instructions(PTO_HEADER)
    header_set = set(header_instrs)
    manifest_set = set(manifest_instrs)

    missing_in_manifest = sorted(header_set - manifest_set)
    if missing_in_manifest:
        errors.append("header instructions missing in manifest: " + ", ".join(missing_in_manifest))

    extra_in_manifest = sorted((manifest_set - header_set) - DOC_ONLY_MANIFEST_INSTRUCTIONS)
    if extra_in_manifest:
        errors.append("manifest instructions not in header: " + ", ".join(extra_in_manifest))

    docs_set = docs_instructions(ISA_DIR)
    missing_en_pages = sorted(manifest_set - docs_set)
    if missing_en_pages:
        errors.append("manifest instructions missing English pages: " + ", ".join(missing_en_pages))

    extra_en_pages = sorted(docs_set - manifest_set)
    if extra_en_pages:
        errors.append("English pages not in manifest: " + ", ".join(extra_en_pages))

    for instr in sorted(manifest_set):
        en_page = ISA_DIR / f"{instr}.md"
        zh_page = ISA_DIR / f"{instr}_zh.md"
        svg = SVG_DIR / f"{instr}.svg"

        if not zh_page.exists():
            errors.append(f"missing zh page: {zh_page}")
        if not svg.exists():
            errors.append(f"missing svg: {svg}")
        if en_page.exists():
            errors.extend(check_en_page(instr, en_page))
        if zh_page.exists():
            errors.extend(check_zh_page(instr, zh_page))

    errors.extend(run_index_check())

    if errors:
        print("ISA consistency check failed:")
        for i, err in enumerate(errors, start=1):
            print(f"  {i}. {err}")
        return 1

    print("OK: ISA consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
