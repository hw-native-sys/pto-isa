# CostModel Backends (stub / fit)

This document explains the two CostModel backends used in this repository: `stub` and `fit`.

## Overview

The CostModel code path is enabled by `__COSTMODEL`, and currently has two backend paths:

- `stub`: baseline CostModel instruction behavior and coverage path
- `fit`: formula-based latency prediction path

Current test suite naming (mapping to backend terms):

- `st` => `stub`
- `st_fit` => `fit`

Related build roots:

- `tests/costmodel/st/`
- `tests/costmodel/st_fit/`

## How They Map in Code

- Shared CostModel entry and types:
  - `include/pto/costmodel/pto_instr.hpp` (used in `stub`)
  - `include/pto/costmodel/lightweight_costmodel.hpp` (used in `fit`)
- Fit-formula backend implementation:
  - `include/pto/costmodel/a2a3/formula_costmodel/formula_backend_compute.hpp`
  - `include/pto/costmodel/a2a3/formula_costmodel/formula_backend_transfer.hpp`

## Stub Backend (Baseline)

- CMake project: `tests/costmodel/st/CMakeLists.txt`
- Focus:
  - Instruction-level CostModel validation
  - Baseline behavior checks for supported ops

## Fit Backend (Formula-Based)

- CMake project: `tests/costmodel/st_fit/CMakeLists.txt`
- Focus:
  - Formula-parameter-based cycle/latency estimation
  - Runtime config impact validation (frequency, bandwidth, tile transfer path)

## Formula Parameter Generation

When running the `fit` backend (`st_fit` suite), the runner generates formula parameter header from CSV:

- Script: `include/pto/costmodel/a2a3/formula_costmodel/gen_formula_params_header.py`
- Input: `include/pto/costmodel/a2a3/formula_costmodel/formula_params.csv`
- Output: `include/pto/costmodel/a2a3/formula_costmodel/formula_params_generated.hpp`

`stub` (`st`) does not require this generation step.

## Run Commands

Run one suite/testcase:

```bash
python3 tests/run_costmodel.py --suite st --testcase tadd --clean --verbose
python3 tests/run_costmodel.py --suite st_fit --testcase time_predict --clean --verbose
```

Batch run both suites (auto-discovery under `st` + `st_fit`):

```bash
bash tests/run_costmodel_tests.sh
```
