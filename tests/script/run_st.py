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

def run_command(command, cwd=None, check=True):
    try:
        print(f"run command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            check=check,
            stdout=None,
            stderr=None,
            text=True
        )
        return ""
    except subprocess.CalledProcessError as e:
        print(f"run command failed with return code {e.returncode}")
        raise

def set_env_variables(run_mode, soc_version):
    if run_mode == "sim":
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_lib_path:
            filtered_paths = [
                path for path in ld_lib_path.split(':')
                if '/runtime/lib64' not in path
            ]
            new_ld_lib = ':'.join(filtered_paths)
            os.environ["LD_LIBRARY_PATH"] = new_ld_lib

        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            raise EnvironmentError("ASCEND_HOME_PATH is not set")

        os.environ["LD_LIBRARY_PATH"] = f"{ascend_home}/runtime/lib64/stub:{os.environ.get('LD_LIBRARY_PATH', '')}"
        if soc_version == "Kirin9030" or soc_version == "KirinX90":
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
                text=True
            )
            for line in result.stdout.splitlines():
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        else:
            print(f"warning: not found {setenv_path}")

        simulator_lib_path = os.path.join(ascend_home, "tools", "simulator", soc_version, "lib")
        os.environ["LD_LIBRARY_PATH"] = f"{simulator_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def build_project(run_mode, soc_version, testcase="all", debug_enable=False, auto_enable=False):
    original_dir = os.getcwd()
    # 清理并创建build目录
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    try:
        cmake_cmd = [
            "cmake",
            f"-DRUN_MODE={run_mode}",
            f"-DSOC_VERSION={soc_version}",
            f"-DTEST_CASE={testcase}",
            ".."
        ]
        if debug_enable :
            cmake_cmd.append("-DDEBUG_MODE=ON")
        if auto_enable:
            cmake_cmd.append("-DAUTO_MODE=ON")

        subprocess.run(
            cmake_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        make_cmd = ["make", "VERBOSE=1"] # print compile log for debug
        # make_cmd = ["make"]
        cpu_count = os.cpu_count() or 4
        make_cmd.extend(["-j", str(cpu_count)])

        result = subprocess.run(
            make_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
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
        cmd = ["cp", golden_path, "build/gen_data.py"]
        run_command(cmd)

        build_dir = "build/"
        os.chdir(build_dir)

        gloden_gen_cmd = [sys.executable, "gen_data.py"]
        output = run_command(gloden_gen_cmd)
        print(output)
    except Exception as e:
        print(f"gen golden failed: {e}")
        raise
    finally:
        os.chdir(original_dir)

RANK_LEVELS = [2, 4, 8]

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

    candidates = [
        "/usr/local/mpich/bin/mpirun",
        "/usr/local/bin/mpirun",
        "/usr/bin/mpirun",
    ]
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
        build_dir = "build/bin/"
        os.chdir(build_dir)

        cmd = ["./" + testcase]
        if args != "all":
            if run_mode == "sim":
                os.environ["CAMODEL_LOG_PATH"] = f"../{args}"
            cmd.append("--gtest_filter=" + args)

        if is_comm:
            mpirun = find_mpirun()
            if not mpirun:
                raise RuntimeError(
                    "mpirun not found. Install MPICH/OpenMPI or set MPI_HOME env.\n"
                    "Also set MPI_LIB_PATH to point to libmpi.so for runtime loading.")
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

        print(f"run command: {' '.join(cmd)}")
        output = run_command(cmd)
        print(output)

    except Exception as e:
        print(f"run binary failed: {e}")
        raise
    finally:
        os.chdir(original_dir)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="执行st脚本")
    parser.add_argument("-r", "--run-mode", required=True, help="运行模式（如 sim or npu)")
    parser.add_argument("-v", "--soc-version", required=True, help="SOC版本 只支持 a3 / a5 / kirin9030 / kirinX90")
    parser.add_argument("-t", "--testcase", required=True, help="需要执行的用例")
    parser.add_argument("-g", "--gtest_filter", required=False, help="可选 需要执行的具体case名")
    parser.add_argument("-d", "--debug-enable", action='store_true', help="开启debug检查")
    parser.add_argument("-a", "--auto-mode-enable", action='store_true', help="开启auto模式")
    parser.add_argument("-w", "--without-build", action='store_true', help="关闭编译（需要预先编译）")
    parser.add_argument("-n", "--nranks", type=int, default=8, help="comm测试的最大MPI rank数量（默认8，自动按2/4/8分轮执行）")

    args = parser.parse_args()
    default_soc_version = "Ascend910B1"
    if args.soc_version == "a5":
        default_soc_version = "Ascend950PR_9599"
    elif args.soc_version == "kirin9030":
        default_soc_version = "Kirin9030"
    elif args.soc_version == "kirinX90":
        default_soc_version = "KirinX90"
    default_cases = "all"
    if args.gtest_filter != None:
        default_cases = args.gtest_filter
    testcase = args.testcase
    is_comm = testcase.startswith("comm/")
    if is_comm:
        testcase = testcase[len("comm/"):]
        if not testcase:
            raise ValueError("comm/ 后必须指定用例名")

    original_dir = os.getcwd()
    try:
        # 获取当前脚本（run_st.py）的绝对路径
        script_path = os.path.abspath(__file__)
        target_dir = os.path.dirname(os.path.dirname(script_path))

        if is_comm and args.soc_version == "a5":
            target_dir = target_dir + "/npu/a5/comm/st"
        elif is_comm:
            target_dir = target_dir + "/npu/a2a3/comm/st"
        elif args.soc_version == "a3":
            target_dir = target_dir + "/npu/a2a3/src/st"
        elif args.soc_version == "kirin9030" or args.soc_version == "kirinX90": # kirin9030 与 kirinX90 共享代码
            target_dir = target_dir + "/npu/kirin9030/src/st"
        else : # a5
            target_dir = target_dir + "/npu/a5/src/st"

        print(f"target_dir: {target_dir}")
        os.chdir(target_dir)

        # 设置环境变量
        set_env_variables(args.run_mode, default_soc_version)

        # 执行构建
        if args.without_build:
            subprocess.run(["rm", "-rf", "build/T*"],
                cwd=original_dir,
                check=True)
        else:
            build_project(args.run_mode, default_soc_version, testcase, args.debug_enable, args.auto_mode_enable)

        # 生成标杆
        golden_path = "testcase/" + testcase + "/gen_data.py"
        run_gen_data(golden_path)

        # 执行二进制文件
        if is_comm and default_cases == "all":
            fail_count = 0
            total_runs = 0
            for nranks in RANK_LEVELS:
                if nranks > args.nranks:
                    continue
                gtest_filter = get_gtest_filter_for_nranks(nranks)
                print(f"============================================================")
                print(f"[INFO] Running comm test: {testcase}  (nranks={nranks}, GTEST_FILTER={gtest_filter})")
                print(f"============================================================")
                os.environ["GTEST_FILTER"] = gtest_filter
                total_runs += 1
                try:
                    run_binary(testcase, args.run_mode, default_cases,
                               is_comm=True, nranks=nranks)
                except Exception as e:
                    print(f"[ERROR] Testcase failed: {testcase} (nranks={nranks})")
                    fail_count += 1
            os.environ.pop("GTEST_FILTER", None)
            print(f"============================================================")
            if fail_count == 0:
                print(f"[INFO] All {total_runs} comm ST run(s) passed.")
            else:
                print(f"[ERROR] {fail_count}/{total_runs} run(s) failed.")
                sys.exit(1)
        else:
            run_binary(testcase, args.run_mode, default_cases,
                       is_comm=is_comm, nranks=args.nranks)

    except Exception as e:
        print(f"run failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    os.chdir(original_dir)

if __name__ == "__main__":
    main()