#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ISA_MANIFEST = REPO_ROOT / "docs" / "isa" / "manifest.yaml"
PUBLIC_HEADERS = [
    REPO_ROOT / "include" / "pto" / "common" / "pto_instr.hpp",
    REPO_ROOT / "include" / "pto" / "comm" / "pto_comm_inst.hpp",
]

DTYPE_PATTERNS = {
    "bool": re.compile(r"\bbool\b"),
    "int8": re.compile(r"\bint8_t\b"),
    "uint8": re.compile(r"\buint8_t\b"),
    "int16": re.compile(r"\bint16_t\b"),
    "uint16": re.compile(r"\buint16_t\b"),
    "int32": re.compile(r"\bint32_t\b"),
    "uint32": re.compile(r"\buint32_t\b"),
    "int64": re.compile(r"\bint64_t\b"),
    "uint64": re.compile(r"\buint64_t\b"),
    "float16": re.compile(r"\bhalf\b"),
    "bfloat16": re.compile(r"\bbfloat16_t\b"),
    "float32": re.compile(r"\bfloat(?:32_t)?\b"),
    "float8_e4m3fn": re.compile(r"\bfloat8_e4m3_t\b"),
    "float8_e5m2": re.compile(r"\bfloat8_e5m2_t\b"),
}
INTRINSIC_PATTERN = re.compile(r"\bPTO_INST\b[^(;\n]*\b(T[A-Z][A-Za-z0-9_]*)\s*\(")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_instruction_manifest() -> list[dict[str, object]]:
    return json.loads(_read(ISA_MANIFEST)).get("instructions", [])


def _public_header_inventory() -> list[str]:
    return [str(path.relative_to(REPO_ROOT)) for path in PUBLIC_HEADERS if path.exists()]


def _intrinsic_inventory() -> list[str]:
    names: set[str] = set()
    for path in PUBLIC_HEADERS:
        if not path.exists():
            continue
        names.update(INTRINSIC_PATTERN.findall(_read(path)))
    return sorted(names)


def _dtype_inventory(text: str) -> dict[str, bool]:
    return {
        name: bool(pattern.search(text))
        for name, pattern in DTYPE_PATTERNS.items()
    }


def build_manifest() -> dict[str, object]:
    instructions = _load_instruction_manifest()
    header_text = "\n".join(_read(path) for path in PUBLIC_HEADERS if path.exists())
    return {
        "source": "pto-isa",
        "instruction_count": len(instructions),
        "instructions": instructions,
        "headers": _public_header_inventory(),
        "intrinsics": _intrinsic_inventory(),
        "frontend_dtypes": _dtype_inventory(header_text),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a PTO-ISA capability manifest")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = build_manifest()
    data = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output is None:
        print(data)
    else:
        args.output.write_text(data + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
