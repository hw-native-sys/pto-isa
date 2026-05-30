"""Build, run, validate, and benchmark shape-specialized GEMM cases.

The runner uses ``kernels/gemm_builder.py`` to generate one kernel for each
selected case. Generated libraries are stored under ``case_builds/<case>/``
and are rebuilt when missing, when the build signature changes, or when
``--rebuild`` is passed.

``compile.sh`` builds the default shared library in ``build_artifacts/``. That
library can also be used with ``--lib`` when exactly one case is selected.
"""

import argparse
import ctypes
import json
import os
import subprocess
from importlib import util as importlib_util
from pathlib import Path

DEVICE_ENV_VAR = "PTODSL_TEST_DEVICE_ID"
TASK_DEVICE_ENV_VAR = "TASK_DEVICE"
DEVICE_PREFIX = "npu:"
DEFAULT_DEVICE_ID = "0"
DEFAULT_NPU_ARCH = "dav-2201"
DEFAULT_SOC = "Ascend910B2"
L2_SAFETY_RATIO = 0.70
L2_STRESS_SAFETY_RATIO = 0.90
L2_STRESS_B_TILE_MIN_BYTES = 32 * 1024 * 1024
GUIDE_SWIZZLE_TARGET_COUNT_N = 5
FP16_BYTES = 2
BUILD_CONFIG_VERSION = 6
torch = None
torch_npu = None

# Cases describe only the shape and base-tile configuration. The runner
# adds SoC-aware launch and L2 swizzle parameters at runtime.
CASE_PRESETS = {
    "baseline_gemm_basic": {
        "description": "GEMM shape 512x2048x1536.",
        "build": {
            "m": 512,
            "k": 2048,
            "n": 1536,
            "baseM": 128,
            "baseK": 64,
            "baseN": 256,
            "stepKa": 4,
            "stepKb": 4,
        },
    },
    "a2a3_allgather_gemm": {
        "description": "Non-square 2048x2048x1024 GEMM.",
        "build": {
            "m": 2048,
            "k": 2048,
            "n": 1024,
            "baseM": 128,
            "baseK": 64,
            "baseN": 256,
            "stepKa": 4,
            "stepKb": 4,
        },
    },
    "a2a3_perf_3072": {
        "description": "Smaller 3072x3072x3072 square GEMM.",
        "build": {
            "m": 3072,
            "k": 3072,
            "n": 3072,
            "baseM": 128,
            "baseK": 64,
            "baseN": 256,
            "stepKa": 4,
            "stepKb": 4,
        },
    },
    "a2a3_allgather_gemm_4096": {
        "description": "AllGather GEMM 4096x4096x4096.",
        "build": {
            "m": 4096,
            "k": 4096,
            "n": 4096,
            "baseM": 128,
            "baseK": 64,
            "baseN": 256,
            "stepKa": 4,
            "stepKb": 4,
        },
    },
    "a2a3_gemm_ar_aligned": {
        "description": "Non-square 5632x6144x1536 GEMM with tall-M shape.",
        "build": {
            "m": 5632,
            "k": 6144,
            "n": 1536,
            "baseM": 128,
            "baseK": 64,
            "baseN": 256,
            "stepKa": 4,
            "stepKb": 4,
        },
    },
    "a2a3_perf_6144": {
        "description": "Square 6144x6144x6144 GEMM.",
        "build": {
            "m": 6144,
            "k": 6144,
            "n": 6144,
            "baseM": 128,
            "baseK": 64,
            "baseN": 256,
            "stepKa": 4,
            "stepKb": 4,
        },
    },
    "a2a3_allgather_gemm_8192": {
        "description": "AllGather GEMM 8192x4096x4096.",
        "build": {
            "m": 8192,
            "k": 4096,
            "n": 4096,
            "baseM": 128,
            "baseK": 64,
            "baseN": 256,
            "stepKa": 4,
            "stepKb": 4,
        },
    },
    "a2a3_allgather_gemm_16384": {
        "description": "AllGather GEMM 16384x4096x4096.",
        "build": {
            "m": 16384,
            "k": 4096,
            "n": 4096,
            "baseM": 128,
            "baseK": 64,
            "baseN": 256,
            "stepKa": 4,
            "stepKb": 4,
        },
    },
    "a2a3_l2_stress_4096_16384_16384": {
        "description": "L2 stress GEMM 4096x16384x16384.",
        "build": {
            "m": 4096,
            "k": 16384,
            "n": 16384,
            "baseM": 128,
            "baseK": 64,
            "baseN": 256,
            "stepKa": 4,
            "stepKb": 4,
        },
    },
}


def get_test_device():
    """Resolve the target NPU device."""
    device_id = os.getenv(DEVICE_ENV_VAR)
    if not device_id:
        task_device = os.getenv(TASK_DEVICE_ENV_VAR)
        if task_device and task_device != "auto":
            device_id = task_device

    if not device_id:
        print(f"Warning: {DEVICE_ENV_VAR} is not set; defaulting to {DEFAULT_DEVICE_ID}.")
        device_id = DEFAULT_DEVICE_ID

    if device_id.startswith(DEVICE_PREFIX):
        return device_id
    return f"{DEVICE_PREFIX}{device_id}"


def ensure_torch_imported():
    """Import torch modules lazily so --list-cases does not require NPU Python."""
    global torch, torch_npu
    if torch is None or torch_npu is None:
        import torch as _torch
        import torch_npu as _torch_npu

        torch = _torch
        torch_npu = _torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def make_inputs(m, k, n, device):
    """Create A, transposed-B storage, and the logical B tensor."""
    a = torch.rand((m, k), device=device, dtype=torch.float16)
    b_ref = torch.rand((k, n), device=device, dtype=torch.float16)
    b_gm = b_ref.transpose(0, 1).contiguous()
    return a, b_gm, b_ref


def resolve_soc_name(cli_soc):
    """Resolve the SoC platform_config name used for L2-aware scheduling."""
    if cli_soc:
        return cli_soc
    for env_name in ("ASCEND_SOC_VERSION", "NPU_SOC_VERSION", "SOC_VERSION", "ASCEND_CHIP_TYPE"):
        soc_name = os.getenv(env_name)
        if soc_name:
            return soc_name
    return DEFAULT_SOC


def platform_config_dir():
    """Find the CANN platform_config directory."""
    candidates = []
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        root = os.getenv(env_name)
        if root:
            candidates.append(Path(root) / "aarch64-linux" / "data" / "platform_config")
            candidates.append(Path(root) / "arm64-linux" / "data" / "platform_config")
    candidates.append(Path("/usr/local/Ascend/cann-9.0.0/aarch64-linux/data/platform_config"))
    candidates.append(Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/data/platform_config"))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def read_platform_int(ini_path, key):
    """Read one integer key from a CANN platform_config ini file."""
    with open(ini_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return int(value.strip(), 0)
    return None


def load_soc_hardware_config(soc_name):
    """Load core count and L2 capacity for the selected SoC."""
    config_dir = platform_config_dir()
    if config_dir is None:
        raise FileNotFoundError("CANN platform_config directory was not found")

    ini_path = config_dir / f"{soc_name}.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"SoC platform config not found: {ini_path}")

    return {
        "soc": soc_name,
        "path": str(ini_path),
        "cube_core_cnt": read_platform_int(ini_path, "cube_core_cnt"),
        "l2_size": read_platform_int(ini_path, "l2_size"),
        "l1_size": read_platform_int(ini_path, "l1_size"),
        "l0_a_size": read_platform_int(ini_path, "l0_a_size"),
        "l0_b_size": read_platform_int(ini_path, "l0_b_size"),
        "l0_c_size": read_platform_int(ini_path, "l0_c_size"),
    }


def _largest_divisor_at_most(value, limit):
    """Return the largest divisor of value not exceeding limit."""
    limit = max(1, min(value, limit))
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def compute_swizzle_count_n(build_cfg, hw_config):
    """Choose the N swizzle width from the SoC L2 budget.

    The scheduler groups adjacent base-N tiles and then walks M rows. The active
    L2 working set is one A base tile plus the grouped B base tiles.
    """
    n_tiles = build_cfg["n"] // build_cfg["baseN"]
    if n_tiles <= 1:
        return 1

    l2_size = hw_config.get("l2_size")
    if not l2_size:
        return 1

    a_tile_bytes = build_cfg["baseM"] * build_cfg["k"] * FP16_BYTES
    b_tile_bytes = build_cfg["baseN"] * build_cfg["k"] * FP16_BYTES
    if b_tile_bytes >= L2_STRESS_B_TILE_MIN_BYTES:
        l2_budget = int(l2_size * L2_STRESS_SAFETY_RATIO)
    else:
        l2_budget = int(l2_size * L2_SAFETY_RATIO)

    available_for_b = l2_budget - a_tile_bytes
    if available_for_b <= 0 or b_tile_bytes <= 0:
        return 1

    l2_max_count_n = available_for_b // b_tile_bytes
    return max(1, min(n_tiles, GUIDE_SWIZZLE_TARGET_COUNT_N, l2_max_count_n))


def effective_build_cfg(case_cfg, hw_config, swizzle_count_n_override=None):
    """Return a case build config with SoC-aware scheduling parameters."""
    build_cfg = dict(case_cfg["build"])
    total_base_tiles = (build_cfg["m"] // build_cfg["baseM"]) * (
        build_cfg["n"] // build_cfg["baseN"]
    )
    cube_core_cnt = hw_config.get("cube_core_cnt") or total_base_tiles
    build_cfg["blockDim"] = min(cube_core_cnt, total_base_tiles)
    if swizzle_count_n_override is None:
        build_cfg["swizzleCountN"] = compute_swizzle_count_n(build_cfg, hw_config)
    else:
        build_cfg["swizzleCountN"] = swizzle_count_n_override
    return build_cfg


def load_lib(lib_path):
    """Load the ctypes launcher exported by caller.cpp."""
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.call_kernel.restype = None

    def gemm_func(c, a, b, block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(c),
            torch_to_ctypes(a),
            torch_to_ctypes(b),
        )

    return gemm_func


def format_shape(build_cfg):
    return f"{build_cfg['m']}x{build_cfg['k']}x{build_cfg['n']}"


def format_case_source(case_cfg):
    return case_cfg.get("source", "kernels/gemm_builder.py")


def validate_build_cfg(case_name, build_cfg):
    m = build_cfg["m"]
    k = build_cfg["k"]
    n = build_cfg["n"]
    base_m = build_cfg["baseM"]
    base_k = build_cfg["baseK"]
    base_n = build_cfg["baseN"]
    block_dim = build_cfg["blockDim"]
    swizzle_count_n = build_cfg.get("swizzleCountN", 1)

    checks = [
        (m % base_m == 0, f"m={m} must be divisible by baseM={base_m}"),
        (k % base_k == 0, f"k={k} must be divisible by baseK={base_k}"),
        (n % base_n == 0, f"n={n} must be divisible by baseN={base_n}"),
        (
            1 <= block_dim <= (m // base_m) * (n // base_n),
            "blockDim must cover the base-tile grid with a persistent loop: "
            f"1 <= blockDim={block_dim} <= "
            f"(m/baseM) * (n/baseN) = "
            f"{(m // base_m) * (n // base_n)}",
        ),
        (
            1 <= swizzle_count_n <= (n // base_n),
            f"swizzleCountN={swizzle_count_n} must be in [1, {n // base_n}]",
        ),
    ]
    for ok, message in checks:
        if not ok:
            raise ValueError(f"Invalid case '{case_name}': {message}")


def load_builder_module(script_dir):
    builder_path = os.path.join(script_dir, "kernels", "gemm_builder.py")
    spec = importlib_util.spec_from_file_location("gemm_builder", builder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load builder module from {builder_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_case_ir(script_dir, output_dir, build_cfg):
    """Generate PTO IR for one case into its case_builds directory."""
    builder = load_builder_module(script_dir)
    ir_module = builder.build(**build_cfg)
    pto_path = os.path.join(output_dir, "gemm.pto")
    with open(pto_path, "w", encoding="utf-8") as f:
        f.write(f"{ir_module}\n")
    return pto_path


def write_case_caller(output_dir):
    """Write the small per-case launcher that includes the generated gemm.cpp."""
    caller_path = os.path.join(output_dir, "caller.cpp")
    caller_code = (
        '#include "gemm.cpp"\n\n'
        'extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *c, uint8_t *a, uint8_t *b)\n'
        "{\n"
        "    Gemm<<<blockDim, nullptr, stream>>>(reinterpret_cast<half *>(c), reinterpret_cast<half *>(a),\n"
        "                                        reinterpret_cast<half *>(b));\n"
        "}\n"
    )
    with open(caller_path, "w", encoding="utf-8") as f:
        f.write(caller_code)
    return caller_path


def case_build_signature(build_cfg):
    """Return the cache signature for generated case artifacts."""
    return {
        "version": BUILD_CONFIG_VERSION,
        "npu_arch": os.getenv("NPU_ARCH", DEFAULT_NPU_ARCH),
        "build": build_cfg,
    }


def case_build_config_path(output_dir):
    return os.path.join(output_dir, "build_config.json")


def write_case_build_signature(output_dir, build_cfg):
    with open(case_build_config_path(output_dir), "w", encoding="utf-8") as f:
        json.dump(case_build_signature(build_cfg), f, indent=2, sort_keys=True)
        f.write("\n")


def case_build_is_current(output_dir, build_cfg):
    try:
        with open(case_build_config_path(output_dir), "r", encoding="utf-8") as f:
            cached = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return cached == case_build_signature(build_cfg)


def build_case_kernel(script_dir, case_name, build_cfg, output_dir):
    """Generate PTO IR, assemble C++, and compile a case shared library."""
    validate_build_cfg(case_name, build_cfg)
    os.makedirs(output_dir, exist_ok=True)
    try:
        pto_path = write_case_ir(script_dir, output_dir, build_cfg)
    except AssertionError as exc:
        raise ValueError(
            f"Case '{case_name}' violates kernels/gemm_builder.py shape constraints: "
            f"{format_shape(build_cfg)} with blockDim={build_cfg['blockDim']}"
        ) from exc
    caller_path = write_case_caller(output_dir)
    cpp_path = os.path.join(output_dir, "gemm.cpp")
    lib_path = os.path.join(output_dir, "gemm.so")
    repo_root = Path(script_dir).parents[2]
    npu_arch = os.getenv("NPU_ARCH", DEFAULT_NPU_ARCH)

    build_cmd = f"""
set -e
if ! command -v bisheng >/dev/null 2>&1; then
    if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
    elif [[ -f /usr/local/Ascend/cann-8.5.0/set_env.sh ]]; then
        source /usr/local/Ascend/cann-8.5.0/set_env.sh || true
    fi
fi

if [[ -n "${{PTOAS_ROOT:-}}" ]]; then
    export PATH="${{PTOAS_ROOT}}/bin:${{PTOAS_ROOT}}:${{PATH}}"
    export LD_LIBRARY_PATH="${{PTOAS_ROOT}}/lib:${{LD_LIBRARY_PATH:-}}"
fi

ptoas --enable-insert-sync "{pto_path}" -o "{cpp_path}"
bisheng -fPIC -shared -xcce -O2 -std=c++17 \\
    --npu-arch="{npu_arch}" \\
    -I{repo_root}/include \\
    "{caller_path}" \\
    -o "{lib_path}"
"""
    print(f"Building case '{case_name}' in {output_dir} ...")
    subprocess.run(["bash", "-c", build_cmd], check=True, cwd=output_dir)
    write_case_build_signature(output_dir, build_cfg)
    return lib_path


def ensure_case_kernel_built(script_dir, case_name, build_cfg, rebuild=False):
    """Return an existing case library or build it when required."""
    output_dir = os.path.join(script_dir, "case_builds", case_name)
    lib_path = os.path.join(output_dir, "gemm.so")
    if rebuild or not os.path.exists(lib_path) or not case_build_is_current(output_dir, build_cfg):
        return build_case_kernel(script_dir, case_name, build_cfg, output_dir)
    return lib_path


def print_available_cases():
    print("Available A2/A3 GEMM cases:")
    for name, case_cfg in CASE_PRESETS.items():
        build_cfg = case_cfg["build"]
        base_shape = f"{build_cfg['baseM']}x{build_cfg['baseK']}x{build_cfg['baseN']}"
        print(
            f"  {name}: {format_shape(build_cfg)}, base={base_shape}, "
            f"source={format_case_source(case_cfg)}"
        )
        print(f"    {case_cfg['description']}")


def correctness_tolerance(k):
    """Absolute tolerance for fp16 output after fp32 accumulation."""
    return max(5e-1, 3e-5 * k)


def test_correctness(gemm_func, m, k, n, block_dim, device):
    """Validate kernel output against torch.matmul computed from logical B."""
    torch.manual_seed(42)

    a, b, b_ref = make_inputs(m, k, n, device)
    c = torch.empty((m, n), device=device, dtype=torch.float16)

    gemm_func(c, a, b, block_dim=block_dim)
    torch.npu.synchronize()

    c_ref = torch.matmul(a, b_ref)
    diff = (c - c_ref).abs().max().item()

    tol = correctness_tolerance(k)
    status = "PASS" if diff <= tol else "FAIL"
    print(
        f"[{status}] m={m}, k={k}, n={n}, block_dim={block_dim}, "
        f"max_diff={diff:.6f} (tol={tol})"
    )
    return diff <= tol


def benchmark(gemm_func, m, k, n, block_dim, device, warmup=5, repeats=20):
    """Measure PTO kernel throughput with ABt layout and NPU event timing."""
    alloc = warmup + repeats
    a_list = [
        torch.randn((m, k), device=device, dtype=torch.float16)
        for _ in range(alloc)
    ]
    b_list = [
        torch.randn((n, k), device=device, dtype=torch.float16)
        for _ in range(alloc)
    ]
    c_list = [
        torch.empty((m, n), device=device, dtype=torch.float16)
        for _ in range(alloc)
    ]

    for c, a, b in zip(c_list[:warmup], a_list[:warmup], b_list[:warmup]):
        gemm_func(c, a, b, block_dim=block_dim)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for c, a, b in zip(c_list[warmup:], a_list[warmup:], b_list[warmup:]):
        gemm_func(c, a, b, block_dim=block_dim)
    end.record()
    torch.npu.synchronize()

    avg_ms = start.elapsed_time(end) / repeats
    tflops = 2.0 * m * k * n / (avg_ms / 1000) / 1e12
    print(f"  [{m}x{k}x{n}] PTO ABt avg={avg_ms:.3f} ms, {tflops:.2f} TFLOPS")
    del a_list, b_list, c_list
    torch.npu.empty_cache()
    return avg_ms


def benchmark_torch_npu(m, k, n, device, warmup=5, repeats=20):
    """Measure torch.matmul with the same ABt input storage and event timing."""
    alloc = warmup + repeats
    a_list = [
        torch.randn((m, k), device=device, dtype=torch.float16)
        for _ in range(alloc)
    ]
    b_list = [
        torch.randn((n, k), device=device, dtype=torch.float16)
        for _ in range(alloc)
    ]

    def matmul_abt(a, b):
        return torch.matmul(a, b.transpose(0, 1))

    for a, b in zip(a_list[:warmup], b_list[:warmup]):
        matmul_abt(a, b)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for a, b in zip(a_list[warmup:], b_list[warmup:]):
        matmul_abt(a, b)
    end.record()
    torch.npu.synchronize()

    avg_ms = start.elapsed_time(end) / repeats
    tflops = 2.0 * m * k * n / (avg_ms / 1000) / 1e12
    print(
        f"  [{m}x{k}x{n}] torch_npu ABt fp16-out "
        f"avg={avg_ms:.3f} ms, {tflops:.2f} TFLOPS"
    )
    del a_list, b_list
    torch.npu.empty_cache()
    return avg_ms


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_lib = os.path.join(script_dir, "build_artifacts", "gemm.so")
    parser = argparse.ArgumentParser(description="GEMM Kernel Test")
    parser.add_argument("--lib", default=default_lib, help="Path to compiled kernel .so")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after correctness test")
    parser.add_argument(
        "--torch-npu",
        action="store_true",
        help="Run torch.matmul timing on NPU with fp16 inputs and fp16 output.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASE_PRESETS.keys()),
        help="A2/A3 GEMM case to build and run. May be passed multiple times.",
    )
    parser.add_argument(
        "--all_cases",
        action="store_true",
        help="Build and run all A2/A3 GEMM cases.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available A2/A3 GEMM cases and exit.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the selected case kernels into case_builds/ before running.",
    )
    parser.add_argument(
        "--soc",
        default=None,
        help=f"SoC platform_config name for L2-aware swizzle (default: {DEFAULT_SOC}).",
    )
    parser.add_argument(
        "--swizzle-count-n",
        "--swizzle-group-m",
        dest="swizzle_count_n",
        type=int,
        default=None,
        help="Override the L2-aware N swizzle width for tuning.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Benchmark warmup iterations.")
    parser.add_argument("--repeats", type=int, default=20, help="Benchmark measurement iterations.")
    args = parser.parse_args()

    if args.list_cases:
        print_available_cases()
        return 0

    if args.all_cases and args.case:
        parser.error("--all_cases cannot be used together with --case")

    selected_cases = list(CASE_PRESETS.keys()) if args.all_cases else args.case or ["a2a3_perf_6144"]
    if args.lib != default_lib and len(selected_cases) != 1:
        parser.error("--lib may only be used when running exactly one case")

    ensure_torch_imported()
    device = get_test_device()
    torch.npu.set_device(device)
    print(f"Using device: {device}")
    soc_name = resolve_soc_name(args.soc)
    hw_config = load_soc_hardware_config(soc_name)
    print(
        f"Using SoC config: {hw_config['soc']} "
        f"cube_core_cnt={hw_config['cube_core_cnt']}, l2_size={hw_config['l2_size']}"
    )
    all_pass = True
    run_benchmark = args.benchmark or args.torch_npu

    for case_name in selected_cases:
        case_cfg = CASE_PRESETS[case_name]
        build_cfg = effective_build_cfg(case_cfg, hw_config, args.swizzle_count_n)
        m = build_cfg["m"]
        k = build_cfg["k"]
        n = build_cfg["n"]
        block_dim = build_cfg["blockDim"]

        print(f"\n=== Case: {case_name} ({format_shape(build_cfg)}) ===")
        print(f"source: {format_case_source(case_cfg)}")
        print(f"swizzleCountN={build_cfg['swizzleCountN']}")

        lib_path = args.lib
        if args.lib == default_lib:
            lib_path = ensure_case_kernel_built(
                script_dir, case_name, build_cfg, rebuild=args.rebuild
            )
        elif args.rebuild:
            parser.error("--rebuild cannot be used together with a custom --lib")

        gemm_func = load_lib(lib_path)

        print("\n=== Correctness Test ===")
        if not test_correctness(gemm_func, m, k, n, block_dim, device):
            all_pass = False

        if run_benchmark:
            print("\n=== Benchmark ===")
            pto_ms = benchmark(
                gemm_func,
                m,
                k,
                n,
                block_dim,
                device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            if args.torch_npu:
                torch_ms = benchmark_torch_npu(
                    m,
                    k,
                    n,
                    device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                print(f"  PTO/torch_npu latency ratio: {torch_ms / pto_ms:.2f}x")

    print(f"\nResult: {'ALL PASS' if all_pass else 'FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
