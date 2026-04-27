# Build and Run

This reference is the execution playbook for PTO-ISA development across Claude Code, Codex, and similar coding agents.

## Documentation and Standards

- Start with:
  - [README.md](../../../../README.md)
  - [docs/agent.md](../../../../docs/agent.md)
  - [docs/getting-started.md](../../../../docs/getting-started.md)
  - [AGENTS.md](../../../../AGENTS.md)
- For docs work, keep markdown concise and unnumbered at the heading level.
- Prefer hub-and-spoke docs:
  - short guidance in a hub doc
  - deep detail in linked reference docs

## CPU-SIM

CPU-SIM is the default path for correctness and portability.

### Why start here

- Works on macOS, Linux, and Windows.
- Does not require Ascend drivers or CANN.
- Gives the fastest loop for instruction semantics and test bring-up.

### Common commands

```bash
python3 tests/run_cpu.py --clean --verbose
python3 tests/run_cpu.py --testcase tadd
python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.*'
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
```

### Notes

- The script auto-detects a suitable compiler.
- BF16 CPU-SIM coverage requires a compiler with `std::bfloat16_t` support.
- The fast CI smoke gate currently uses `tadd`; the full gate uses `tests/run_cpu_tests.sh`.
- For pipe, split-lane, layout, or backend-template changes, pair the focused testcase with the matching review gate in [review-derived-guardrails.md](review-derived-guardrails.md).

## Costmodel

Use the costmodel runner when the change affects predicted cost rather than execution semantics.

```bash
python3 tests/run_costmodel.py --testcase tcolmax --build-dir build/costmodel_tcolmax --generator Ninja --clean
python3 tests/run_costmodel.py --testcase tcolsum --build-dir build/costmodel_tcolsum --generator Ninja --clean
python3 tests/run_costmodel.py --testcase tmrgsort --build-dir build/costmodel_tmrgsort --generator Ninja --clean
python3 tests/run_costmodel.py --testcase ttrans --build-dir build/costmodel_ttrans --generator Ninja --clean
```

## NPU Simulator and Real NPU

Use [tests/script/run_st.py](../../../../tests/script/run_st.py) for both Ascend simulator and hardware ST.

### Single-test pattern

```bash
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
python3 tests/script/run_st.py -r npu -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
```

### Wrapper script

```bash
chmod +x tests/run_st.sh
./tests/run_st.sh --a3 --sim --simple
./tests/run_st.sh --a5 --npu --simple
```

### Environment requirements

- NPU ST is Linux-oriented.
- `ASCEND_HOME_PATH` must be set.
- Source the toolkit environment before running:

```bash
source /usr/local/Ascend/cann/set_env.sh
```

or the installed equivalent described in [docs/getting-started.md](../../../../docs/getting-started.md).

## Cross-Platform Strategy

To keep work portable:

- Author and validate semantics on CPU-SIM first.
- Avoid backend-specific tile locations unless the feature explicitly requires them.
- Treat A2/A3 and A5 as separate conformance targets, not just separate compile flags.
- Use [docs/mkdocs/src/manual/12-backend-profiles-and-conformance.md](../../../../docs/mkdocs/src/manual/12-backend-profiles-and-conformance.md) plus [include/README.md](../../../../include/README.md) to decide whether a behavior is meant to be portable.

## Regression Closure Checklist

Before calling a targeted test addition complete:

- the testcase directory exists and has its local `CMakeLists.txt`
- the parent `CMakeLists.txt` registers the target
- generated/golden data distinguishes the layouts or offsets under test
- the gtest filter matches a real test name
- stale interface names were searched with `rg` when an enum, overload, or public signature changed
- backend-specific code has an A2/A3 or A5 smoke command when local hardware/sim access is available

## Documentation Build

When you change docs, build them locally:

```bash
cmake -S docs -B build/docs
cmake --build build/docs --target pto_docs
```

The generated site lands in `build/docs/site/`.

For ISA manual taxonomy or bilingual navigation changes, prefer the stricter MkDocs path used in recent doc PRs when the local environment has it available:

```bash
python -m mkdocs build -f docs/mkdocs/mkdocs.yml --strict
```
