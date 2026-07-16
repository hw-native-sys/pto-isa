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


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def check_required_pages(errors: list[str]) -> None:
    for rel in REQUIRED_PAGES:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"missing required PTO ISA page: {rel}")


def check_nav(errors: list[str]) -> None:
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
