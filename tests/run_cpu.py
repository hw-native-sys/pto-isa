#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import argparse
import os
import re
import shutil
import subprocess
import sys
import site
import time
import logging
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def _format_cmd(command: List[str]) -> str:
    return " ".join(map(str, command))


def run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    *,
    title: Optional[str] = None,
    verbose: bool = False,
    always_print_patterns: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> float:
    cwd_str = str(cwd) if cwd is not None else None
    start = time.perf_counter()
    if title:
        logging.info(f"{title}")
    if verbose:
        logging.info(f"  $ {_format_cmd(command)}" + (f"\n  cwd: {cwd_str}" if cwd_str else ""))
    try:
        completed = subprocess.run(
            [str(x) for x in command],
            cwd=cwd_str,
            env=env or os.environ,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0 or verbose:
            if completed.stdout:
                logging.info(completed.stdout.rstrip())
            if completed.stderr:
                logging.info(completed.stderr.rstrip())
        elif always_print_patterns:
            patterns = [re.compile(p) for p in always_print_patterns]
            for line in (completed.stdout or "").splitlines():
                if any(p.search(line) for p in patterns):
                    logging.info(line.rstrip())
            for line in (completed.stderr or "").splitlines():
                if any(p.search(line) for p in patterns):
                    logging.info(line.rstrip())
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )
    except FileNotFoundError as e:
        raise RuntimeError(f"command not found: {command[0]}") from e
    return time.perf_counter() - start


def add_user_scripts_to_path() -> None:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return
        major, minor = sys.version_info[:2]
        scripts_dir = Path(appdata) / "Python" / f"Python{major}{minor}" / "Scripts"
    else:
        scripts_dir = Path(site.getuserbase()) / "bin"

    if scripts_dir.exists():
        scripts_str = str(scripts_dir)
        current_path = os.environ.get("PATH", "")
        if scripts_str not in current_path.split(os.pathsep):
            os.environ["PATH"] = scripts_str + os.pathsep + current_path


def ensure_cmake_tools() -> None:
    if shutil.which("cmake") and shutil.which("ctest"):
        return
    logging.info("cmake/ctest not found, installing via pip...")
    run_command([sys.executable, "-m", "pip", "install", "--user", "cmake>=3.16"])
    add_user_scripts_to_path()
    if not (shutil.which("cmake") and shutil.which("ctest")):
        raise RuntimeError(
            "cmake/ctest still not found after installation; please add user Scripts/bin directory to PATH"
        )


def is_windows() -> bool:
    if os.name == "nt" or platform.system().lower() == "windows":
        return True
    return False


def cmake_friendly_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    if is_windows():
        p = p.replace("\\", "/")
    return p


def get_compiler_major_version(compiler_path: str) -> int:
    """Get the major version number of the compiler."""
    if not compiler_path:
        return 0

    try:
        logging.debug("Checking version for compiler: %s", compiler_path)
        # check=False ensures that even if the command returns a non-zero status code,
        # it will not raise CalledProcessError, but judge by result.returncode.
        result = subprocess.run(
            [compiler_path, "--version"],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logging.warning("Failed to run --version on: %s", compiler_path)
            return 0

        match = re.search(r'(\d+)\.', result.stdout)
        if match:
            version = int(match.group(1))
            logging.debug("Parsed version for %s: %d", compiler_path, version)
            return version

    except Exception as e:
        logging.warning("Exception occurred while checking compiler version: %s", e)
        return 0

    return 0


def _try_find_compiler(cxx_name: str, cc_name: str, min_ver: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to find a specific C++ compiler and check if the version meets the requirements.
    """
    cxx_path = shutil.which(cxx_name)
    if not cxx_path:
        return None, None

    ver = get_compiler_major_version(cxx_path)

    # Log the detection result
    logging.debug("Found candidate %s, version: %d (required: %d)", cxx_path, ver, min_ver)

    if ver >= min_ver:
        cc_path = shutil.which(cc_name)
        logging.info("Selected compiler pair: %s / %s (Version >= %d)", cxx_path, cc_path, min_ver)
        return cxx_path, cc_path

    return None, None


def _auto_detect_compilers() -> Tuple[str, Optional[str]]:
    logging.info("CXX not specified, starting automatic detection...")

    # 1. Try Clang
    cxx, cc = _try_find_compiler("clang++", "clang", 15)
    if cxx:
        return cxx, cc

    # 2. Try GCC
    cxx, cc = _try_find_compiler("g++", "gcc", 13)
    if cxx:
        return cxx, cc

    # 3. Fail
    error_msg = (
        "Could not find a suitable compiler.\n"
        "Requirements:\n"
        " - clang++ >= 15\n"
        " - OR g++ >= 13"
    )
    logging.error(error_msg)
    raise RuntimeError(error_msg)


def _derive_cc_from_cxx(cxx_path: str) -> Optional[str]:
    """
    Guess the corresponding CC based on the path name of CXX.
    """
    if not cxx_path:
        return None

    logging.debug("Attempting to derive CC from CXX: %s", cxx_path)
    name = Path(cxx_path).name

    # Match as long as the path contains "g++" or "clang" keywords
    if "clang" in name:
        logging.info("Derived CC as clang")
        return shutil.which("clang")

    if "g++" in name:
        logging.info("Derived CC as gcc")
        return shutil.which("gcc")

    return None


def detect_compilers(cxx_arg: Optional[str], cc_arg: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    # 1. Initialize variables
    cxx = cxx_arg or os.environ.get("CXX")
    cc = cc_arg or os.environ.get("CC")

    if cxx:
        logging.info("Using explicit CXX: %s", cxx)

    # 2. Determine CXX path
    if not cxx:
        # Auto detection mode
        cxx, auto_cc = _auto_detect_compilers()
        if not cc:
            cc = auto_cc
    elif not Path(cxx).is_absolute():
        # Resolve relative path
        resolved_cxx = shutil.which(cxx)
        if resolved_cxx:
            logging.debug("Resolved relative path '%s' to '%s'", cxx, resolved_cxx)
            cxx = resolved_cxx

    # 3. Determine CC path
    if not cc:
        cc = _derive_cc_from_cxx(cxx)
    elif not Path(cc).is_absolute():
        resolved_cc = shutil.which(cc) or cc
        if resolved_cc != cc:
            logging.debug("Resolved relative path '%s' to '%s'", cc, resolved_cc)
        cc = resolved_cc

    # 4. Format paths
    if cxx:
        cxx = cmake_friendly_path(cxx)
    if cc:
        cc = cmake_friendly_path(cc)

    logging.info("Final Compiler Selection -> CXX: %s, CC: %s", cxx, cc)
    return cxx, cc


def cmake_build(build_dir: Path, build_type: str) -> None:
    cmd: List[str] = ["cmake", "--build", str(build_dir), "--parallel"]

    # Multi-config generators (e.g. VS) need --config; single-config ignores it.
    cmd.extend(["--config", build_type])
    run_command(cmd)


def generate_golden(build_dir: Path, gen_script: Path) -> None:
    dst = build_dir / "gen_data.py"
    st_dir = gen_script.resolve().parent.parent.parent
    shutil.copyfile(gen_script, dst)

    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    new_path = str(st_dir)
    env["PYTHONPATH"] = f"{new_path}{os.pathsep}{pp}" if pp else new_path

    run_command([sys.executable, str(dst.name)], cwd=build_dir, env=env)


def read_cmake_cache_var(build_dir: Path, var_name: str) -> Optional[str]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    try:
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith(("//", "#")):
                continue
            # CMakeCache format: VAR:TYPE=VALUE
            if line.startswith(f"{var_name}:"):
                _, _, value = line.partition("=")
                return value
    except OSError:
        return None
    return None


def find_binaries(build_dir: Path, build_type: str) -> Dict[str, Path]:
    bin_dir = build_dir / "bin"
    if os.name == "nt":
        config_dir = bin_dir / build_type
        if config_dir.exists():
            bin_dir = config_dir

    if not bin_dir.exists():
        return {}

    binaries: Dict[str, Path] = {}
    for p in bin_dir.iterdir():
        if not p.is_file():
            continue
        if os.name == "nt":
            if p.suffix.lower() != ".exe":
                continue
            binaries[p.stem] = p
        else:
            binaries[p.name] = p
    return binaries


def run_gtest_binary(binary: Path, gtest_filter: Optional[str], xml_output: Optional[Path], build_type: str,
                     verbose: bool) -> None:
    cmd: List[str] = [str(binary)]
    if gtest_filter:
        cmd.append(f"--gtest_filter={gtest_filter}")
    if xml_output:
        xml_output.parent.mkdir(parents=True, exist_ok=True)
        cmd.append(f"--gtest_output=xml:{xml_output}")

    # CPU ST test data is under build_dir/..., and tests use paths like "../<suite.case>/input1.bin".
    # For multi-config generators on Windows, binaries are under build/bin/<Config>/, so we run from build/bin/.
    run_cwd = binary.parent
    if os.name == "nt" and binary.parent.name.lower() == build_type.lower():
        run_cwd = binary.parent.parent
    run_command(cmd, cwd=run_cwd, verbose=verbose)


def run_binary(binary: Path, build_type: str, cwd: Optional[Path] = None) -> None:
    run_cwd = cwd or binary.parent
    if os.name == "nt" and binary.parent.name.lower() == build_type.lower():
        run_cwd = binary.parent.parent
    run_command([str(binary)], cwd=run_cwd)


def build_and_run_demo(demo_name: str, repo_root: Path, build_type: str, cxx: Optional[str], cc: Optional[str], *,
	                   verbose: bool) -> None:
    demos_root = repo_root / ".." / "demos" / "cpu"
    demo_map: dict[str, tuple[Path, str]] = {
        "gemm": (demos_root / "gemm_demo", "gemm_demo"),
        "flash_attn": (demos_root / "flash_attention_demo", "flash_attention_demo"),
        "mla": (demos_root / "mla_attention_demo", "mla_attention_demo"),
    }
    if demo_name not in demo_map:
        raise RuntimeError(f"unknown demo: {demo_name}")

    demo_src, exe_stem = demo_map[demo_name]
    legacy_demo_src = repo_root / "demo"
    if demo_name == "gemm" and not demo_src.exists() and legacy_demo_src.exists():
        demo_src = legacy_demo_src
        exe_stem = "gemm_demo"
    if not demo_src.exists():
        raise RuntimeError(f"demo dir not found: {demo_src}")

    demo_build = demo_src / "build"
    if demo_build.exists():
        shutil.rmtree(demo_build)
    demo_build.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "cmake",
            "-S",
            str(demo_src),
            "-B",
            str(demo_build),
            f"-DCMAKE_BUILD_TYPE={build_type}",
            *([f"-DCMAKE_C_COMPILER={cc}"] if cc else []),
            *([f"-DCMAKE_CXX_COMPILER={cxx}"] if cxx else []),
        ],
        title="[STEP] demo: cmake configure",
        verbose=verbose,
    )
    run_command(
        ["cmake", "--build", str(demo_build), "--parallel", "--config", build_type],
        title="[STEP] demo: cmake build",
        verbose=verbose,
    )

    exe_name = f"{exe_stem}.exe" if os.name == "nt" else exe_stem
    exe = demo_build / exe_name
    if os.name == "nt":
        exe = demo_build / build_type / exe_name

    if not exe.exists():
        raise RuntimeError(f"demo binary not found: {exe}")

    run_command([str(exe)],
                cwd=(exe.parent.parent
                    if (os.name == "nt" and exe.parent.name.lower() == build_type.lower())
                    else exe.parent),
                title=f"[STEP] demo: run {exe_stem}",
                verbose=verbose,
                always_print_patterns=[r"^perf:"])


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cols: List[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out = [sep, fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    out.append(sep)
    return "\n".join(out)


def _parse_duration_seconds(s: str) -> float:
    if s.endswith("ms"):
        return float(s[:-2]) / 1000.0
    if s.endswith("s"):
        return float(s[:-1])
    return 0.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build & run CPU simulator ST unit tests (tests/cpu/st)",
        epilog=("Examples:\n  python run_cpu.py --build-type Release\n"
            "  python run_cpu.py --testcase tadd --build-type Release\n"
            "  python run_cpu.py --no-build --gtest_filter TADDTest.*\n"
            "  python run_cpu.py --demo gemm\n"
            "  python run_cpu.py --demo flash_attn\n"
            "  python run_cpu.py --demo mla\n"
            "  python run_cpu.py --demo all\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="Show full output from cmake/msbuild/gtest (default: \
                        quiet, only structured logs).",)
    parser.add_argument("-t", "--testcase", help="Run a single testcase (e.g. tadd). Default: run all built bin.",)
    parser.add_argument("-g", "--gtest_filter", help="Optional gtest filter (e.g. 'TADDTest.case1').",)
    parser.add_argument("--cxx", help="C++ compiler (e.g. clang++). Default: $CXX or auto-detect.")
    parser.add_argument("--cc", help="C compiler (e.g. clang). Default: $CC or auto-detect.")
    parser.add_argument("--build-type", default="Release", choices=["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
                        help="CMake build type.",)
    parser.add_argument("--build-dir", default=None, help="Build directory. Default: tests/cpu/st/build",)
    parser.add_argument("--no-clean", action="store_true", help="(Deprecated) No-op; kept for backward compatibility.")
    parser.add_argument("--clean", action="store_true", help="Delete build dir and rebuild.")
    parser.add_argument("--rebuild", action="store_true", help="Force re-configure and rebuild .")
    parser.add_argument("--no-build", action="store_true", help="Skip cmake configure/build, only run existing bin")
    parser.add_argument("--no-gen", action="store_true", help="Skip running testcase gen_data.py.")
    parser.add_argument("--xml-dir", default=None, help="If set, write gtest xml reports under this directory")
    parser.add_argument("--no-install", action="store_true", help="Do not auto-install missing tools/deps (numpy).")
    parser.add_argument("--demo", choices=["gemm", "flash_attn", "mla", "all"], default=None, help="Build & run demo program \
                        (e.g. 'gemm', 'flash_attn'). \
                        Note: demo runs alone (does not run CPU ST).")
    parser.add_argument("--demo-only", action="store_true", help="Same as --demo (demo runs without CPU ST).")
    parser.add_argument("--generator", default=None, help="CMake generator(Windows required: 'MinGW Makefiles' etc..)")
    parser.add_argument("--cmake_prefix_path", default=None, help="-DCMAKE_PREFIX_PATH=<path> e.g. D:\\gtest")
    parser.add_argument(
        "--enable-bf16",
        action="store_true",
        help="Enable BF16 CPU-SIM coverage. Requires a compiler with C++23 std::bfloat16_t support.",
    )
    args = parser.parse_args()
    return args


def setup_environment(args) -> None:
    if not args.no_install:
        add_user_scripts_to_path()
        ensure_cmake_tools()


def resolve_bf16_compiler_pair(args) -> None:
    from tests.script.cpu_bfloat16 import detect_bfloat16_cxx, derive_cc_from_cxx

    selected_cxx = detect_bfloat16_cxx(args.cxx)
    if args.cxx and (shutil.which(args.cxx) or args.cxx) != selected_cxx:
        raise RuntimeError(f"--cxx={args.cxx} does not support std::bfloat16_t")
    args.cxx = selected_cxx
    if not args.cc:
        args.cc = derive_cc_from_cxx(selected_cxx)


def log_build_info(args, cxx, cc) -> None:
    logging.info(f"[INFO] build_type={args.build_type}")
    logging.info(f"[INFO] bf16={'ON' if args.enable_bf16 else 'OFF'}")
    if cxx:
        logging.info(f"[INFO] cxx={cxx}")
    if cc:
        logging.info(f"[INFO] cc={cc}")


def run_demo_mode(args, repo_root, cxx, cc) -> int:
    if args.demo_only and not args.demo:
        logging.error("error: --demo-only requires --demo")
        return 2
    demo_name = args.demo or "gemm"
    if not args.demo:
        # should not happen due to demo_name assignment, but keep the guard
        pass

    logging.info("\n== DEMO ==")
    demos = ["gemm", "flash_attn", "mla"] if demo_name == "all" else [demo_name]
    t0 = time.perf_counter()
    for name in demos:
        build_and_run_demo(
            demo_name=name, repo_root=repo_root, build_type=args.build_type, cxx=cxx, cc=cc, verbose=args.verbose
        )
    demo_time = time.perf_counter() - t0
    logging.info(f"[PASS] demo: {demo_name} ({_format_seconds(demo_time)})")
    return 0


def run_test_mode(args, repo_root, cxx, cc) -> int:
    source_dir = repo_root / "cpu" / "st"
    if not source_dir.exists():
        logging.error(f"error: not found CPU ST dir: {source_dir}")
        return 2

    build_dir = Path(args.build_dir) if args.build_dir else (source_dir / "build")
    if not build_dir.is_absolute():
        build_dir = (repo_root / build_dir).resolve()

    if args.clean:
        if build_dir.exists():
            shutil.rmtree(build_dir)

    need_build = determine_need_build(args, source_dir, build_dir)
    if need_build:
        if not perform_build(args, source_dir, build_dir, cxx, cc):
            return 2
    else:
        logging.info("\n== BUILD ==")
        logging.info("[SKIP] build (already built)")
    return execute_tests(args, source_dir, build_dir)


def parse_expected_testcases(source_dir: Path) -> Optional[set[str]]:
    cmake_list = source_dir / "testcase" / "CMakeLists.txt"
    if not cmake_list.exists():
        return None

    text = cmake_list.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"set\(ALL_TESTCASES\s*(.*?)\)", text, flags=re.DOTALL)
    if not m:
        return None

    body = m.group(1)
    cases: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        cases.extend(line.split())
    return set(cases)


def determine_need_build(args, source_dir: Path, build_dir: Path) -> bool:
    binaries_before = find_binaries(build_dir, args.build_type) if build_dir.exists() else {}
    configured_testcase = read_cmake_cache_var(build_dir, "TEST_CASE") if build_dir.exists() else None
    config_mismatch = False
    if args.testcase:
        if configured_testcase != args.testcase:
            config_mismatch = True
    else:
        # If the existing build was configured to only build a single testcase,
        # force a reconfigure so "run all" truly builds all.
        if configured_testcase:
            config_mismatch = True

    have_requested_binary = True
    if args.testcase:
        have_requested_binary = args.testcase in binaries_before
    else:
        have_requested_binary = bool(binaries_before)
        expected = parse_expected_testcases(source_dir)
        if expected:
            missing = expected.difference(binaries_before.keys())
            if missing:
                have_requested_binary = False

    need_build = (
        (not args.no_build)
        and (
            config_mismatch
            or args.rebuild
            or args.clean
            or not (build_dir / "CMakeCache.txt").exists()
            or not have_requested_binary
        )
    )
    return need_build


def perform_build(args, source_dir, build_dir, cxx, cc) -> bool:
    build_dir.mkdir(parents=True, exist_ok=True)
    logging.info("\n== BUILD ==")
    if is_windows() and not args.generator:
        logging.error("On Windows, must specify --generator (\"MinGW Makefiles\" or \"Ninja\", etc..)")
        return False
    cfg_time = run_command(
        [
            "cmake",
            *([] if args.testcase else ["-UTEST_CASE"]),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={args.build_type}",
            f"-DPTO_CPU_SIM_ENABLE_BF16={'ON' if args.enable_bf16 else 'OFF'}",
            *([f"-DTEST_CASE={args.testcase}"] if args.testcase else []),
            *([f"-DCMAKE_C_COMPILER={cc}"] if cc else []),
            *([f"-DCMAKE_CXX_COMPILER={cxx}"] if cxx else []),
            *(["-G", args.generator] if args.generator else []),
            *([f"-DCMAKE_PREFIX_PATH={args.cmake_prefix_path}"] if args.cmake_prefix_path else []),
        ],
        title="[STEP] cmake configure",
        verbose=args.verbose,
    )
    build_time = run_command(
        ["cmake", "--build", str(build_dir), "--parallel", "--config", args.build_type],
        title="[STEP] cmake build",
        verbose=args.verbose,
    )
    logging.info(f"[PASS] build ({_format_seconds(cfg_time + build_time)})")
    return True


def execute_tests(args, source_dir, build_dir) -> int:
    binaries = find_binaries(build_dir, args.build_type)
    if not binaries:
        logging.error(f"error: no binaries found under {build_dir / 'bin'} (did build succeed?)")
        return 2

    selected: list[tuple[str, Path]]
    if args.testcase:
        if args.testcase not in binaries:
            known = ", ".join(sorted(binaries.keys()))
            logging.error(f"error: unknown testcase '{args.testcase}'. Built binaries: {known}")
            return 2
        selected = [(args.testcase, binaries[args.testcase])]
    else:
        selected = sorted(binaries.items(), key=lambda x: x[0])

    xml_dir: Optional[Path] = None
    if args.xml_dir:
        xml_dir = Path(args.xml_dir)
        if not xml_dir.is_absolute():
            xml_dir = build_dir / xml_dir

    results = run_selected_tests(args, source_dir, build_dir, selected, xml_dir)
    if results:
        print_test_summary(results)
    return 0


def run_selected_tests(args, source_dir, build_dir, selected, xml_dir) -> List[List[str]]:
    logging.info("\n== TESTS ==")
    results: List[List[str]] = []
    for testcase, binary in selected:
        gen_script = source_dir / "testcase" / testcase / "gen_data.py"
        if not args.no_gen and gen_script.exists():
            logging.info(f"[STEP] gen_data: {testcase}")
            old_pythonpath = os.environ.get("PYTHONPATH", "")
            old_bf16_flag = os.environ.get("PTO_CPU_SIM_ENABLE_BF16")
            repo_root = Path(__file__).resolve().parent.parent
            os.environ["PYTHONPATH"] = str(repo_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
            if args.enable_bf16:
                os.environ["PTO_CPU_SIM_ENABLE_BF16"] = "1"
            else:
                os.environ.pop("PTO_CPU_SIM_ENABLE_BF16", None)
            try:
                generate_golden(build_dir=build_dir, gen_script=gen_script)
            finally:
                os.environ["PYTHONPATH"] = old_pythonpath
                if old_bf16_flag is None:
                    os.environ.pop("PTO_CPU_SIM_ENABLE_BF16", None)
                else:
                    os.environ["PTO_CPU_SIM_ENABLE_BF16"] = old_bf16_flag

        xml_output = (xml_dir / f"{testcase}.xml") if xml_dir else None
        t0 = time.perf_counter()
        try:
            run_gtest_binary(
                binary=binary,
                gtest_filter=args.gtest_filter,
                xml_output=xml_output,
                build_type=args.build_type,
                verbose=args.verbose
            )
            status = "PASS"
        except Exception:
            status = "FAIL"
            raise
        finally:
            elapsed = time.perf_counter() - t0
            results.append([testcase, status, _format_seconds(elapsed)])
            logging.info(f"[{status}] {testcase} ({_format_seconds(elapsed)})")
    return results


def print_test_summary(results) -> None:
    logging.info("\n== SUMMARY ==")
    total_time_s = sum(_parse_duration_seconds(r[2]) for r in results)
    results.append(["TOTAL", "", _format_seconds(total_time_s)])
    logging.info(_render_table(["Target", "Status", "Time"], results))


def main() -> int:
    args = parse_arguments()
    setup_environment(args)
    repo_root = Path(__file__).resolve().parent

    if args.enable_bf16:
        resolve_bf16_compiler_pair(args)

    cxx, cc = detect_compilers(args.cxx, args.cc)
    log_build_info(args, cxx, cc)

    if args.demo or args.demo_only:
        return run_demo_mode(args, repo_root, cxx, cc)

    return run_test_mode(args, repo_root, cxx, cc)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    raise SystemExit(main())
