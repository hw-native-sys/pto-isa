# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import sys
import subprocess
import shutil
import argparse
import fnmatch
import glob
import re


RE_GTEST_RUNNING = re.compile(r"\[==========\] Running (\d+) tests? from \d+ test suite")


def run_command(command, cwd=None, check=True):
    try:
        print(f"run command: {' '.join(command)}")
        result = subprocess.run(command, cwd=cwd, check=check, stdout=None, stderr=None, text=True)
        return ""
    except subprocess.CalledProcessError as e:
        print(f"run command failed with return code {e.returncode}")
        raise


def run_gtest_binary(command, cwd=None):
    """Run a gtest binary, relay its output line by line, and return the number
    of test cases it reported running.

    Output is relayed instead of inherited so the count can be read from the
    same stream the CI log captures. A gtest binary that matches no test still
    exits 0, so callers must treat a zero count as a failure: a mistyped
    --gtest_filter would otherwise drop a whole suite while CI stays green.
    """
    print(f"run command: {' '.join(command)}")
    proc = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    ran = 0
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        match = RE_GTEST_RUNNING.search(line)
        if match:
            ran += int(match.group(1))
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return ran


def expected_case_count(gtest_filter):
    """Number of cases a fully literal gtest_filter should run, or None if the
    filter uses wildcards / negative patterns and the count is not predictable."""
    if gtest_filter in (None, "", "all", "*"):
        return None
    if any(ch in gtest_filter for ch in "*?-"):
        return None
    return len([p for p in gtest_filter.split(":") if p])


def count_tests_in_binary(binary):
    """Count the cases a binary registers, via --gtest_list_tests.

    gtest prints the list and exits before any environment SetUp runs, so this
    costs a process start but no aclInit. Used for the unfiltered invocations
    where the expected count cannot be derived from the command line.
    """
    try:
        result = subprocess.run(
            [binary, "--gtest_list_tests"], capture_output=True, text=True, timeout=120
        )
    except Exception as e:
        print(f"[CASES] --gtest_list_tests failed for {binary}: {e}")
        return None
    if result.returncode != 0:
        print(f"[CASES] --gtest_list_tests exited {result.returncode} for {binary}")
        return None
    count = 0
    for line in result.stdout.splitlines():
        # Suite lines are unindented and end with '.'; case lines are indented.
        if line and not line[0].isspace() and line.rstrip().endswith("."):
            continue
        if line.strip() and line[0].isspace():
            count += 1
    return count


def set_env_variables(run_mode, soc_version):
    if run_mode == "sim":
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_lib_path:
            filtered_paths = [path for path in ld_lib_path.split(":") if "/runtime/lib64" not in path]
            new_ld_lib = ":".join(filtered_paths)
            os.environ["LD_LIBRARY_PATH"] = new_ld_lib

        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            raise EnvironmentError("ASCEND_HOME_PATH is not set")

        os.environ["LD_LIBRARY_PATH"] = f"{ascend_home}/runtime/lib64/stub:{os.environ.get('LD_LIBRARY_PATH', '')}"
        if soc_version == "Kirin9030" or soc_version == "KirinX90" or soc_version == "Kirin9050":
            setenv_path = os.path.join(ascend_home, "set_env.sh")
        else:
            setenv_path = os.path.join(ascend_home, "bin", "setenv.bash")
        if os.path.exists(setenv_path):
            print(f"run env shell: {setenv_path}")
            result = subprocess.run(
                f"source {setenv_path} && env",
                shell=True,
                executable=shutil.which("bash") or "bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        else:
            print(f"warning: not found {setenv_path}")

        _, simulator_lib_path = get_simulator_info(ascend_home, soc_version)
        os.environ["LD_LIBRARY_PATH"] = f"{simulator_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def get_simulator_info(ascend_home, soc_version):
    simulator_home = os.path.join(ascend_home, "tools", "simulator")
    soc_candidates = [soc_version]
    if soc_version == "Ascend950PR_9599":
        soc_candidates.extend(["Ascend910_9599"])
    for candidate in soc_candidates:
        camodel_path = os.path.join(simulator_home, candidate, "camodel")
        lib_path = os.path.join(simulator_home, candidate, "lib")
        if os.path.isdir(camodel_path):
            return candidate, camodel_path
        elif os.path.isdir(lib_path):
            return candidate, lib_path
    print(f"Warning: Neither 'camodel' nor 'lib' found in {os.path.join(simulator_home, soc_version)}")
    return soc_version, os.path.join(simulator_home, soc_version, "lib")


def build_project(run_mode, soc_version, testcase="all", debug_enable=False, auto_enable=False):
    original_dir = os.getcwd()
    # 清理并创建build目录（可通过 PTO_BUILD_DIR 环境变量覆盖，用于多卡并行隔离）
    build_dir = os.environ.get("PTO_BUILD_DIR", "build")
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # Resolve SOC_VERSION for simulator path (e.g. Ascend950PR_9599 -> Ascend910_9599)
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if run_mode == "sim" and ascend_home:
        cmake_soc, _ = get_simulator_info(ascend_home, soc_version)
    else:
        cmake_soc = soc_version

    try:
        cmake_cmd = ["cmake", f"-DRUN_MODE={run_mode}", f"-DSOC_VERSION={cmake_soc}", f"-DTEST_CASE={testcase}", ".."]
        if debug_enable:
            cmake_cmd.append("-DDEBUG_MODE=ON")
        if auto_enable:
            cmake_cmd.append("-DAUTO_MODE=ON")

        subprocess.run(
            cmake_cmd, cwd=build_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        make_cmd = ["make", "VERBOSE=1"]  # print compile log for debug
        # make_cmd = ["make"]
        cpu_count = os.cpu_count() or 4
        make_cmd.extend(["-j", str(cpu_count)])

        result = subprocess.run(
            make_cmd, cwd=build_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        print("compile process:\n", result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"build failed: {e.stdout}")
        raise
    finally:
        os.chdir(original_dir)


def run_gen_data(golden_path):
    original_dir = os.getcwd()
    try:
        build_dir = os.environ.get("PTO_BUILD_DIR", "build")
        # Use a testcase-specific filename to avoid races when multiple
        # gen_data invocations target the same build_dir in parallel.
        _tc_name = os.path.basename(os.path.dirname(golden_path))
        _gen_script = f"gen_data_{_tc_name}.py"
        cmd = ["cp", golden_path, f"{build_dir}/{_gen_script}"]
        run_command(cmd)

        os.chdir(build_dir)

        gloden_gen_cmd = [sys.executable, _gen_script]
        output = run_command(gloden_gen_cmd)
        print(output)
    except Exception as e:
        print(f"gen golden failed: {e}")
        raise
    finally:
        os.chdir(original_dir)


def needs_test_isolation(testcase):
    """CCU tests need process isolation (one mpirun per GTest case)."""
    return testcase.endswith("_ccu")


def list_gtest_cases(testcase_dir, gtest_filter="*"):
    """Parse TEST_F macros from source — no binary execution, no device access."""
    main_path = os.path.join("testcase", testcase_dir, "main.cc")
    try:
        with open(main_path) as f:
            content = f.read()
    except FileNotFoundError:
        return []
    pairs = re.findall(r"^\s*TEST_F\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", content, re.MULTILINE)
    tests = [f"{suite}.{name}" for suite, name in pairs]
    if "-" in gtest_filter:
        pos, neg = gtest_filter.split("-", 1)
        neg_patterns = [p for p in neg.split(":") if p]
        tests = [t for t in tests if not any(fnmatch.fnmatch(t, p) for p in neg_patterns)]
    elif gtest_filter != "*":
        patterns = [p for p in gtest_filter.split(":") if p]
        tests = [t for t in tests if any(fnmatch.fnmatch(t, p) for p in patterns)]
    return tests


RANK_LEVELS = [2, 4, 8]


def detect_npu_count():
    """Count available NPU devices on this host.

    Probes /dev/davinci0, /dev/davinci1, ... (excluding davinci_manager/devmm_svm).
    Returns the count, or None if /dev/davinci* devices are not present (likely
    running on a host without NPUs — let downstream checks decide).
    """
    pattern = re.compile(r"^/dev/davinci(\d+)$")
    devs = [p for p in glob.glob("/dev/davinci*") if pattern.match(p)]
    if not devs:
        return None
    return len(devs)


def get_gtest_filter_for_nranks(nranks):
    """Build GTEST_FILTER based on test naming convention (*_NRanks / *_Nranks)."""
    if nranks == 2:
        return "*-*4Ranks*:*4ranks*:*8Ranks*:*8ranks*"
    elif nranks == 4:
        return "*4Ranks*:*4ranks*"
    elif nranks == 8:
        return "*8Ranks*:*8ranks*"
    return "*"


def find_mpirun():
    """Find mpirun executable, checking MPI_HOME and common paths."""
    mpi_home = os.environ.get("MPI_HOME", "")
    if mpi_home:
        candidate = os.path.join(mpi_home, "bin", "mpirun")
        if os.path.isfile(candidate):
            return candidate

    candidates = ["/usr/local/mpich/bin/mpirun", "/usr/local/bin/mpirun", "/usr/bin/mpirun"]
    for c in candidates:
        if os.path.isfile(c):
            return c

    result = shutil.which("mpirun")
    if result:
        return result

    return None


def run_binary(testcase, run_mode, args="all", is_comm=False, nranks=2):
    original_dir = os.getcwd()
    try:
        build_dir = os.environ.get("PTO_BUILD_DIR", "build") + "/bin/"
        os.chdir(build_dir)

        if run_mode == "sim":
            camodel_log_dir = "camodel_log"
            os.makedirs("log/ub_log", exist_ok=True)
            os.makedirs(camodel_log_dir, exist_ok=True)
            os.environ["CAMODEL_LOG_PATH"] = camodel_log_dir

        cmd = ["./" + testcase]
        if args != "all":
            cmd.append("--gtest_filter=" + args)

        if is_comm:
            mpirun = find_mpirun()
            if not mpirun:
                raise RuntimeError(
                    "mpirun not found. Install MPICH/OpenMPI or set MPI_HOME env.\n"
                    "Also set MPI_LIB_PATH to point to libmpi.so for runtime loading."
                )
            mpi_cmd = [mpirun, "-n", str(nranks)]
            try:
                ver = subprocess.run([mpirun, "--version"], capture_output=True, text=True)
                ver_text = ver.stdout + ver.stderr
                if "open mpi" in ver_text.lower() or "openmpi" in ver_text.lower():
                    mpi_cmd.append("--allow-run-as-root")
            except Exception:
                pass
            cmd = mpi_cmd + cmd
            mpi_lib_dir = os.path.dirname(mpirun).replace("/bin", "/lib")
            if os.path.isdir(mpi_lib_dir):
                os.environ["MPI_LIB_PATH"] = os.path.join(mpi_lib_dir, "libmpi.so")

        # Case-count guard is opt-in: run_st.sh enables it by exporting
        # PTO_CASE_COUNT_FILE. Other callers (A3/Kirin/comm) keep the original
        # behaviour.
        count_file = None if is_comm else os.environ.get("PTO_CASE_COUNT_FILE")

        if count_file is None:
            print(f"run command: {' '.join(cmd)}")
            output = run_command(cmd)
            print(output)
        else:
            ran = run_gtest_binary(cmd)
            expected = expected_case_count(args)
            if expected is None and args == "all":
                expected = count_tests_in_binary("./" + testcase)
            print(f"[CASES] {testcase} ran={ran} expected={expected if expected is not None else 'n/a'}")
            # One short append per invocation; O_APPEND keeps concurrent
            # writers from interleaving, so run_st.sh can sum these later.
            with open(count_file, "a") as _f:
                _f.write(f"{ran} {expected if expected is not None else -1} {testcase}\n")
            if ran == 0:
                raise RuntimeError(
                    f"{testcase} ran 0 test cases (gtest_filter={args!r}). "
                    "The filter matched nothing, so this suite would be silently skipped."
                )
            if expected is not None and ran != expected:
                raise RuntimeError(
                    f"{testcase} ran {ran} test cases but the filter lists {expected} "
                    f"(gtest_filter={args!r}). At least one case name does not match."
                )

    except Exception as e:
        print(f"run binary failed: {e}")
        raise
    finally:
        os.chdir(original_dir)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="执行st脚本")
    parser.add_argument("-r", "--run-mode", required=True, help="运行模式（如 sim or npu)")
    parser.add_argument(
        "-v", "--soc-version", required=True, help="SOC版本 只支持 a3 / a5 / a6 / kirin9030 / kirinX90 kirin9050"
    )
    parser.add_argument("-t", "--testcase", required=True, help="需要执行的用例")
    parser.add_argument("-g", "--gtest_filter", required=False, help="可选 需要执行的具体case名")
    parser.add_argument("-d", "--debug-enable", action="store_true", help="开启debug检查")
    parser.add_argument("-a", "--auto-mode-enable", action="store_true", help="开启auto模式")
    parser.add_argument("-w", "--without-build", action="store_true", help="关闭编译（需要预先编译）")
    parser.add_argument(
        "-n", "--nranks", type=int, default=8, help="comm测试的最大MPI rank数量（默认8，自动按2/4/8分轮执行）"
    )

    args = parser.parse_args()
    default_soc_version = "Ascend910B1"
    if args.soc_version == "a5":
        default_soc_version = "Ascend950PR_9599"
    elif args.soc_version == "kirin9030":
        default_soc_version = "Kirin9030"
    elif args.soc_version == "kirinX90":
        default_soc_version = "KirinX90"
    elif args.soc_version == "kirin9050":
        default_soc_version = "Kirin9050"
    elif args.soc_version == "a6":
        default_soc_version = "dav_9201"
    default_cases = "all"
    if args.gtest_filter != None:
        default_cases = args.gtest_filter
    testcase = args.testcase
    is_comm = testcase.startswith("comm/")
    if is_comm:
        testcase = testcase[len("comm/") :]
        if not testcase:
            raise ValueError("comm/ 后必须指定用例名")

    original_dir = os.getcwd()
    try:
        # 获取当前脚本（run_st.py）的绝对路径
        script_path = os.path.abspath(__file__)
        target_dir = os.path.dirname(os.path.dirname(script_path))

        if is_comm and args.soc_version == "a5":
            target_dir = target_dir + "/npu/a5/comm/st"
        elif is_comm and args.soc_version == "a6":
            target_dir = target_dir + "/npu/a6/comm/st"
        elif is_comm:
            target_dir = target_dir + "/npu/a2a3/comm/st"
        elif args.soc_version == "a3":
            target_dir = target_dir + "/npu/a2a3/src/st"
        elif args.soc_version == "a6":
            target_dir = target_dir + "/npu/a6/src/st"
        elif args.soc_version == "kirin9030":  # kirin9030 与 kirinX90 共享代码
            target_dir = target_dir + "/npu/kirin9030/src/st"
        elif args.soc_version == "kirinX90":
            target_dir = target_dir + "/npu/kirinX90/src/st"
        elif args.soc_version == "kirin9050":
            target_dir = target_dir + "/npu/kirin9050/src/st"
        else:  # a5
            target_dir = target_dir + "/npu/a5/src/st"

        print(f"target_dir: {target_dir}")
        os.chdir(target_dir)

        # 设置环境变量
        set_env_variables(args.run_mode, default_soc_version)

        # 执行构建
        if args.without_build:
            # _build_dir is relative to target_dir (we chdir'd there above), the
            # same base every other PTO_BUILD_DIR use relies on. Expand the
            # pattern here: subprocess does not go through a shell, so "T*"
            # would reach rm as a literal path and -f would swallow the miss.
            _build_dir = os.environ.get("PTO_BUILD_DIR", "build")
            for _stale in glob.glob(os.path.join(_build_dir, "T*")):
                subprocess.run(["rm", "-rf", _stale], check=True)
        else:
            build_project(args.run_mode, default_soc_version, testcase, args.debug_enable, args.auto_mode_enable)

        # 生成标杆（a6 共享用例无本地目录，回落到 a5孪生用例的 gen_data.py）
        golden_path = "testcase/" + testcase + "/gen_data.py"
        if not os.path.exists(golden_path):
            golden_path = "../../../a5/src/st/testcase/" + testcase + "/gen_data.py"
        run_gen_data(golden_path)

        # 执行二进制文件
        if is_comm and default_cases == "all":
            fail_count = 0
            total_runs = 0
            isolated = needs_test_isolation(testcase)
            available_npus = detect_npu_count()
            for nranks in RANK_LEVELS:
                if nranks > args.nranks:
                    continue
                if available_npus is not None and nranks > available_npus:
                    print(f"[SKIP] {testcase} (nranks={nranks}): only {available_npus} NPU(s) available")
                    continue
                gtest_filter = get_gtest_filter_for_nranks(nranks)

                if isolated:
                    # CCU tests: run each GTest case in a separate mpirun
                    cases = list_gtest_cases(testcase, gtest_filter)
                    if not cases:
                        print(f"[WARN] No tests discovered for {testcase} (nranks={nranks})")
                        continue
                    os.environ.pop("GTEST_FILTER", None)
                    for case in cases:
                        print("============================================================")
                        print(f"[INFO] Running comm test: {testcase} / {case}  (nranks={nranks}, isolated)")
                        print("============================================================")
                        total_runs += 1
                        try:
                            run_binary(testcase, args.run_mode, case, is_comm=True, nranks=nranks)
                        except Exception:
                            print(f"[ERROR] Testcase failed: {testcase}/{case} (nranks={nranks})")
                            fail_count += 1
                else:
                    print("============================================================")
                    print(f"[INFO] Running comm test: {testcase}  (nranks={nranks}, GTEST_FILTER={gtest_filter})")
                    print("============================================================")
                    os.environ["GTEST_FILTER"] = gtest_filter
                    total_runs += 1
                    try:
                        run_binary(testcase, args.run_mode, default_cases, is_comm=True, nranks=nranks)
                    except Exception:
                        print(f"[ERROR] Testcase failed: {testcase} (nranks={nranks})")
                        fail_count += 1
            os.environ.pop("GTEST_FILTER", None)
            print("============================================================")
            if fail_count == 0:
                print(f"[INFO] All {total_runs} comm ST run(s) passed.")
            else:
                print(f"[ERROR] {fail_count}/{total_runs} run(s) failed.")
                sys.exit(1)
        else:
            run_binary(testcase, args.run_mode, default_cases, is_comm=is_comm, nranks=args.nranks)

    except Exception as e:
        print(f"run failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    os.chdir(original_dir)


if __name__ == "__main__":
    main()
