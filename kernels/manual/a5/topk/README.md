# TopK (PTO SIMD VF)

A5 TopK demo: `ITER_SELECT` (iterative top-1) for small `N/K`, `RADIX_SELECT` (asc-devkit-style radix) for large `N/K`. Kernel lives in `TopK.hpp`; gtests in `main.cpp` / `topk_test_kernel.cpp`.

## Prerequisites

```bash
source ~/source_cann.sh   # sets ASCEND_HOME_PATH
```

Default SoC: `Ascend950PR_9599` (camodel).

## Run tests

Wrapper: `./run.sh` (cmake + make + gtest; camodel dumps go under `build/<case>/`).

```bash
# From this directory (pto-isa/dv300_exp/TopK)

# List known case names
./run.sh -l

# Single case (sim, default SoC)
./run.sh -c radix_1x4096_k512
./run.sh -c iter_1x128_k3

# Reuse existing build/ (skip cmake)
./run.sh -k -c iter_1x256_k8

# All wired cases
./run.sh

# Help
./run.sh -h
```

Dumps after a run: `build/<case>/core*_summary_log`, `*.dump`, `run.log`.

### Wired gtest cases

rvec = `rvec_busy_cycle` from A5 camodel (`perf/<case>/`, via `python3 parse_perf_table.py`).

| Case | Algo | Shape | rvec cycles |
|------|------|-------|------------:|
| `iter_1x128_k3` | ITER_SELECT | N=128, K=3 | 124 |
| `iter_1x256_k8` | ITER_SELECT | N=256, K=8 | 352 |
| `iter_1x256_k12` | ITER_SELECT | N=256, K=12 | 483 |
| `iter_1x288_k8` | ITER_SELECT | N=288, K=8 | 426 |
| `iter_1x384_k6` | ITER_SELECT | N=384, K=6 | 392 |
| `iter_1x512_k12` | ITER_SELECT | N=512, K=12 | 921 |
| `iter_1x896_k16` | ITER_SELECT | N=896, K=16 | 2,035 |
| `radix_1x4096_k512` | RADIX_SELECT | N=4096, K=512 | 8,668 |
| `radix_1x16384_k256` | RADIX_SELECT | N=16384, K=256 | 12,353 |
| `radix_1x16384_k2048` | RADIX_SELECT | N=16384, K=2048 | 19,408 |

Golden inputs live under `TopKTest.radix_*/` (`input.bin`, `golden_value.bin`, `golden_index.bin`).

## Collect / parse performance

Batch collect into `perf/<case>/` (needs a built `build/topk`):

```bash
# Build once if needed
./run.sh -c iter_1x128_k3

# Collect camodel logs for cases listed in collect_perf.sh
bash collect_perf.sh
# bash collect_perf.sh --quick   # skip the heaviest iter case

# Parse rvec / scalar / mte / kernel ticks → table + perf/summary.json
python3 parse_perf_table.py
```

`parse_perf_table.py` reads `perf/<case>/core*_summary_log` and `core*.veccore*_su_perf_summary_log`.

## Layout

| Path | Role |
|------|------|
| `TopK.hpp` | Device TopK implementation |
| `topk_test_kernel.cpp` | Kernel launch / explicit instantiations |
| `main.cpp` | gtest cases |
| `run.sh` | Build + run + dump |
| `collect_perf.sh` | Batch camodel collect → `perf/` |
| `parse_perf_table.py` | Perf table + `perf/summary.json` |
| `TopKTest.radix_*/` | Golden binaries |
