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

"""Check PTO ISA manual structure and MkDocs source hygiene.

The canonical PTO ISA manual now lives under docs/isa/. MkDocs owns only
landing pages and assets under docs/mkdocs/src/; generated mirrors must not be
tracked there.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MKDOCS_YML = REPO_ROOT / "docs" / "mkdocs" / "mkdocs.yml"

REQUIRED_PAGES = [
    "docs/isa/README.md",
    "docs/isa/introduction/document-structure.md",
    "docs/isa/programming-model/tiles-and-valid-regions.md",
    "docs/isa/machine-model/execution-agents.md",
    "docs/isa/syntax-and-operands/assembly-model.md",
    "docs/isa/state-and-types/type-system.md",
    "docs/isa/memory-model/consistency-baseline.md",
    "docs/isa/instruction-families/README.md",
    "docs/isa/tile/README.md",
    "docs/isa/vector/README.md",
    "docs/isa/scalar/README.md",
    "docs/isa/comm/README.md",
    "docs/isa/system/README.md",
    "docs/isa/reference/README.md",
]

EXPECTED_NAV_SECTIONS = [
    "1. Introduction",
    "2. Programming Model",
    "3. Machine Model",
    "4. Syntax And Operands",
    "5. State And Types",
    "6. Memory Model",
    "7. Instruction Set Overview",
    "8. Tile Instruction Reference",
    "9. Vector Instruction Reference",
    "10. Scalar And Control Reference",
    "11. Communication ISA Reference",
    "12. System Scheduling ISA Reference",
    "13. Reference Notes",
]

ALLOWED_MKDOCS_SRC_EXACT = {
    "docs/mkdocs/src/index.md",
    "docs/mkdocs/src/index_zh.md",
}

ALLOWED_MKDOCS_SRC_PREFIXES = ("docs/mkdocs/src/assets/",)

REMOVED_PTOAS_PAGES = [
    "docs/assembly/PTO-AS.md",
    "docs/assembly/PTO-AS_zh.md",
    "docs/assembly/PTO-AS.bnf",
]

LINK_SCAN_ROOTS = [
    "docs",
    "tests/npu/a5/src/st/testcase/mgather",
    "tests/npu/a5/src/st/testcase/mscatter",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def check_required_pages(errors: list[str]) -> None:
    for rel in REQUIRED_PAGES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required PTO ISA page: {rel}")


def check_nav(errors: list[str]) -> None:
    text = read(MKDOCS_YML)
    last = -1
    for section in EXPECTED_NAV_SECTIONS:
        needle = f"- {section}:"
        idx = text.find(needle)
        if idx == -1:
            errors.append(f"missing MkDocs nav section: {section}")
            continue
        if idx < last:
            errors.append(f"MkDocs nav section out of order: {section}")
        last = idx

    if "PTO-AS Reference" in text:
        errors.append("MkDocs nav must not expose PTO-AS as a standalone reference section")
    if "docs/assembly/PTO-AS.md" in text or "docs/assembly/PTO-AS_zh.md" in text:
        errors.append("MkDocs nav must route PTO-AS through docs/isa/syntax-and-operands/assembly-model*.md")
    if "docs/isa/syntax-and-operands/assembly-model.md" not in text:
        errors.append("MkDocs nav must include the PTO-AS spelling page under Syntax And Operands")
    if "docs/isa/other/" in text or "other-families" in text or "Non-ISA" in text:
        errors.append("MkDocs nav must use Tile, Vector, Scalar, Communication, and System Scheduling only")


def check_isa_source_of_truth(errors: list[str]) -> None:
    text = read(REPO_ROOT / "docs" / "isa" / "README.md")
    required = [
        "This directory is the canonical PTO ISA tree",
        "Textual assembly spelling belongs to the PTO ISA syntax instruction set",
        "not a separate ISA with different semantics",
    ]
    for snippet in required:
        if snippet not in text:
            errors.append(f"docs/isa/README.md is missing source-of-truth statement: {snippet}")


def check_ptoas_single_location(errors: list[str]) -> None:
    for rel in REMOVED_PTOAS_PAGES:
        if (REPO_ROOT / rel).exists():
            errors.append(f"duplicate PTO-AS manual source remains: {rel}")

    forbidden = ("assembly/PTO-AS", "docs/assembly/PTO-AS", "PTO-AS.bnf")
    for root_rel in LINK_SCAN_ROOTS:
        root = REPO_ROOT / root_rel
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            text = read(path)
            for needle in forbidden:
                if needle in text:
                    errors.append(
                        "PTO-AS manual links must point to "
                        f"docs/isa/syntax-and-operands/assembly-model*.md: {path.relative_to(REPO_ROOT)}"
                    )
                    break


def check_mkdocs_source_hygiene(errors: list[str]) -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "docs/mkdocs/src"],
        cwd=REPO_ROOT,
        text=True,
    ).splitlines()
    leaked = [
        rel
        for rel in tracked
        if (REPO_ROOT / rel).exists()
        and rel not in ALLOWED_MKDOCS_SRC_EXACT
        and not any(rel.startswith(prefix) for prefix in ALLOWED_MKDOCS_SRC_PREFIXES)
    ]
    if leaked:
        errors.append(
            "tracked generated or mirrored MkDocs source files remain:\n"
            + "\n".join(f"  - {rel}" for rel in leaked[:100])
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PTO ISA docs hierarchy and MkDocs source hygiene")
    parser.add_argument("--check", action="store_true", help="compatibility flag; checks are always performed")
    _ = parser.parse_args()

    errors: list[str] = []
    check_required_pages(errors)
    check_nav(errors)
    check_isa_source_of_truth(errors)
    check_ptoas_single_location(errors)
    check_mkdocs_source_hygiene(errors)

    if errors:
        print("PTO ISA documentation consistency check failed:")
        for idx, err in enumerate(errors, 1):
            print(f"  {idx}. {err}")
        return 1

    print("OK: PTO ISA documentation hierarchy is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
