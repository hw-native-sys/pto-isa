# tests/script/

Entry scripts for building and running the repository test suites.

## NPU ST (sim / npu)

- Build + run: `tests/script/run_st.py`
- Build only: `tests/script/build_st.py`

Common arguments:

- `-r, --run-mode`: `sim` or `npu`
- `-v, --soc-version`: `a3` or `a5` (mapped to an internal `SOC_VERSION`)
- `-t, --testcase`: testcase name (e.g., `tmatmul`)
- `-g, --gtest_filter`: optional gtest filter (run a single case)
- `-d, --debug-enable`: optional debug build (only in `run_st.py`)

Examples:

```bash
python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
```

## CPU ST

- Batch build + run: `tests/script/all_cpu_tests.py`

Options:

- `-v, --verbose`: print build/run output
- `-b, --build-folder`: build directory (default: `build_tests`)

Example:

```bash
python3 tests/script/all_cpu_tests.py --verbose
```

## Convenience Wrappers

- Recommended suites: `tests/run_st.sh`
- CPU tests: `tests/run_cpu_tests.sh`

For the latest arguments, run `python3 <script> -h`.
