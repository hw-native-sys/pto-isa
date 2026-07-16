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

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import venv


REQUIRED_MODULES = ["mkdocs", "mkdocs_gen_files", "pymdownx"]
REQUIREMENTS_TXT = Path(__file__).resolve().parent / "requirements.txt"


def _missing_modules_in_current_interpreter() -> list[str]:
    return [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]


def _missing_modules_for(python_exe: Path) -> list[str]:
    code = (
        "import importlib.util\n"
        f"req={REQUIRED_MODULES!r}\n"
        "miss=[m for m in req if importlib.util.find_spec(m) is None]\n"
        "print('\\n'.join(miss))\n"
        "raise SystemExit(1 if miss else 0)\n"
    )
    try:
        proc = subprocess.run([str(python_exe), "-c", code], check=False, capture_output=True, text=True)
    except OSError:
        return REQUIRED_MODULES[:]
    if proc.returncode == 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv(venv_dir: Path) -> Path:
    venv_python = _venv_python(venv_dir)
    if not venv_python.exists():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(str(venv_dir))
    return venv_python


def _pip_install_requirements(python_exe: Path) -> None:
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS_TXT)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--ensure-venv",
        metavar="DIR",
        type=Path,
        help="Create/update a virtualenv and install docs requirements into it.",
    )
    args = parser.parse_args(argv)

    if args.ensure_venv is not None:
        try:
            venv_python = _ensure_venv(args.ensure_venv)
            missing = _missing_modules_for(venv_python)
            if missing:
                _pip_install_requirements(venv_python)
                missing = _missing_modules_for(venv_python)
            if missing:
                sys.stderr.write("error: missing Python packages in venv: " + ", ".join(missing) + "\n")
                sys.stderr.write(f"venv: {args.ensure_venv}\n")
                sys.stderr.write(f"python: {venv_python}\n")
                sys.stderr.write(f"requirements: {REQUIREMENTS_TXT}\n")
                return 1
            return 0
        except Exception as e:  # noqa: BLE001 - best-effort tool bootstrap with actionable output
            sys.stderr.write(f"error: failed to bootstrap docs venv: {e}\n")
            sys.stderr.write(f"venv: {args.ensure_venv}\n")
            sys.stderr.write(f"requirements: {REQUIREMENTS_TXT}\n")
            return 1

    missing = _missing_modules_in_current_interpreter()
    if not missing:
        return 0

    sys.stderr.write("error: missing Python packages: " + ", ".join(missing) + "\n")
    sys.stderr.write(f"python: {sys.executable}\n")
    sys.stderr.write(f"install: python -m pip install -r {REQUIREMENTS_TXT.as_posix()}\n")
    sys.stderr.write("tip: prefer a venv and point CMake at it via -DPython3_EXECUTABLE=<venv>/bin/python\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
