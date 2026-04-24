import ctypes
import os
import subprocess
from pathlib import Path
import torch
from util.bench import do_bench
from util.device import get_test_device

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]
DEVICE = get_test_device()
torch.npu.set_device(DEVICE)

ROWS = 32
COLS = 256
FULL_ROWS = 128
SLOTS = 4
SINGLE = 0
CHUNKED = 1
WARMUP = 10
BENCH = 30
ITERS = 4096


HERE = Path(__file__).resolve().parent
SO = HERE / "tpush_chunk_bench.so"
subprocess.run(
    [
        "bisheng",
        "-fPIC",
        "-shared",
        "-xcce",
        f"--npu-arch={os.environ.get('NPU_ARCH', 'dav-2201').strip()}",
        "-O2",
        "-std=c++17",
        "-Wno-ignored-attributes",
        f"-I{PTO_LIB_PATH}/include",
        f"-I{PTO_LIB_PATH}/kernels/manual/a2a3/flash_atten",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        str(HERE / "tpush_chunk_bench.cpp"),
        "-o",
        str(SO),
    ],
    check=True,
)

run = ctypes.CDLL(os.path.abspath(SO)).call_kernel
run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
stream = torch.npu.current_stream()._as_parameter_
ptr = ctypes.c_void_p


def call(x: torch.Tensor, y: torch.Tensor, fifo: torch.Tensor, mode: int):
    run(stream, ptr(x.data_ptr()), ptr(y.data_ptr()), ptr(fifo.data_ptr()), mode, ITERS)


def make():
    x = torch.randn((ROWS, COLS), device=DEVICE, dtype=torch.float16)
    return x, torch.empty_like(x), torch.empty((SLOTS, FULL_ROWS, COLS), device=DEVICE, dtype=torch.float16)


x0, y0, f0 = make()
x1, y1, f1 = make()
x1.copy_(x0)

call(x0, y0, f0, SINGLE)
call(x1, y1, f1, CHUNKED)
torch.npu.synchronize()

torch.testing.assert_close(y0, x0)
torch.testing.assert_close(y1, x1)
torch.testing.assert_close(y0, y1)

t0 = do_bench(lambda: call(x0, y0, f0, SINGLE), warmup_iters=WARMUP, benchmark_iters=BENCH, unit="ms")
t1 = do_bench(lambda: call(x1, y1, f1, CHUNKED), warmup_iters=WARMUP, benchmark_iters=BENCH, unit="ms")

print(f"kernel_iters : {ITERS}")
print(f"single mode  : {t0:.3f} ms")
print(f"chunked mode : {t1:.3f} ms")
print(f"speedup      : {t0 / t1:.3f}x")
print(f"time saved   : {100.0 * (t0 - t1) / t0:.1f}%")

SO.unlink(missing_ok=True)
