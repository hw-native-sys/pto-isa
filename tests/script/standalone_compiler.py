# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Assess the actual compiler used by GNU-style standalone CPU suites.

Only positive identification of an unsupported driver permits an aggregate
skip. A failed query or GNU capability probe is an error. The sole usage-query
exception is MSVC's rejection of --version: its banner must be confirmed by
a successful native /? query with CL options before classifying its driver.
"""

import json
from pathlib import Path
import re
import subprocess


def _as_text(chunk):
    return chunk if isinstance(chunk, str) else (chunk or b"").decode(errors="replace")


def assess_compiler(compiler, output_dir, cwd, timeout=10, compiler_args=()):
    """Assess the CPU CMake standard policy, retaining every query and its output."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assessment = {
        "compiler": str(compiler),
        "status": "error",
        "family": "unknown",
        "driver": "unknown",
        "reason": "compiler assessment did not complete",
        "probes": [],
    }

    def finish(status, reason):
        assessment.update(status=status, reason=reason)
        (output_dir / "compiler.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
        return assessment

    def probe(name, flags, stdin=None):
        argv = [str(compiler), *map(str, compiler_args), *map(str, flags)]
        record = {
            "name": name,
            "argv": argv,
            "cwd": str(cwd),
            "exit": None,
            "timed_out": False,
            "launch_error": None,
            "log_path": str(output_dir / (name + ".log")),
        }
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout,
            )
            text = proc.stdout + proc.stderr
            record["exit"] = proc.returncode
        except subprocess.TimeoutExpired as exc:
            text = _as_text(exc.stdout) + _as_text(exc.stderr)
            record["timed_out"] = True
        except OSError as exc:
            text = f"{type(exc).__name__}: {exc}\n"
            record["launch_error"] = text.rstrip()
        Path(record["log_path"]).write_text(text, encoding="utf-8")
        assessment["probes"].append(record)
        return record["exit"] == 0 and not record["timed_out"] and record["launch_error"] is None, text

    def failed(name):
        return finish("error", f"compiler {name} probe failed; see {assessment['probes'][-1]['log_path']}")

    ok, version = probe("compiler-version", ["--version"])
    version_query = assessment["probes"][-1]
    if version_query["timed_out"] or version_query["launch_error"] is not None:
        return failed("version")
    if re.search(r"Microsoft\s*\(R\)\s*C/C\+\+", version, re.IGNORECASE):
        assessment.update(family="msvc", driver="cl")
        # cl commonly prints its banner but rejects --version and no source.
        # Confirm native help, rather than interpreting any failed probe as
        # evidence of incompatibility or relying on the executable's basename.
        help_ok, help_text = probe("compiler-driver", ["/?"])
        if not help_ok:
            return failed("MSVC driver identification")
        cl_options = set(re.findall(r"^\s*/(D|I|Fe|std)(?=\b|<|:)", help_text, re.MULTILINE))
        if not ({"D", "I"} <= cl_options or "std" in cl_options):
            return finish("error", "Microsoft compiler banner was not confirmed by native CL help options")
        return finish("unsupported", "MSVC's CL driver is outside this standalone suite's GNU-style driver contract")
    if not ok:
        return failed("version")

    is_clang = bool(re.search(r"\bclang version\b", version, re.IGNORECASE))
    checked_clang_driver = False

    def check_clang_driver():
        nonlocal checked_clang_driver
        # -help succeeds in both Clang driver modes; --help fails in clang-cl.
        # A Windows target or _MSC_VER alone does not identify the CL driver.
        ok, help_text = probe("compiler-driver", ["-help"])
        checked_clang_driver = True
        if not ok:
            return failed("driver identification")
        if "CL.EXE COMPATIBILITY OPTIONS:" in help_text:
            assessment.update(family="clang", driver="cl")
            return finish(
                "unsupported", "Clang's CL driver is outside this standalone suite's GNU-style driver contract"
            )
        return None

    if is_clang:
        assessment["family"] = "clang"
        result = check_clang_driver()
        if result is not None:
            return result

    ok, macros = probe("compiler-macros", ["-dM", "-E", "-x", "c++", "-"], stdin="")
    if not ok:
        return failed("predefined macros")
    clang_macro = bool(re.search(r"^#define\s+__clang__\b", macros, re.MULTILINE))
    gnu_macro = bool(re.search(r"^#define\s+__GNUC__\b", macros, re.MULTILINE))
    if clang_macro:
        assessment["family"] = "clang"
        if not checked_clang_driver:
            result = check_clang_driver()
            if result is not None:
                return result
    elif gnu_macro and not is_clang:
        assessment["family"] = (
            "gcc"
            if re.search(r"Free Software Foundation|\bgcc\b|\bg\+\+\b", version, re.IGNORECASE)
            else "gnu-compatible"
        )
    else:
        return finish("error", "compiler identity is unknown or inconsistent; no confirmed GNU/Clang macro evidence")

    assessment["driver"] = "gnu"
    gnu_major = re.search(r"^#define\s+__GNUC__\s+(\d+)\s*$", macros, re.MULTILINE)
    if assessment["family"] == "gcc" and gnu_major is None:
        return finish("error", "GNU compiler major version is unknown")
    # Match CPU CMake: only positively identified GNU>=14 needs C++23.
    standard = 23 if assessment["family"] == "gcc" and int(gnu_major[1]) >= 14 else 20
    standard_flag = f"-std=c++{standard}"
    assessment.update(standard=standard, standard_flag=standard_flag)
    minimum = 202100 if standard == 23 else 202002
    source = output_dir / "compiler-probe.cpp"
    source.write_text(
        f"#if __cplusplus < {minimum}L\n#error C++{standard} mode was not selected\n#endif\nint main() {{ return 0; }}\n",
        encoding="utf-8",
    )
    ok, _ = probe(f"compiler-cpp{standard}", [standard_flag, "-fsyntax-only", str(source)])
    if not ok:
        return failed(f"GNU C++{standard} syntax")
    return finish("supported", f"GNU-style compiler identity and independent C++{standard} syntax probe succeeded")
