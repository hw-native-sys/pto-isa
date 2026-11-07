# TADD ST Testcase — Requirements

## Purpose

Verify the `TADD` instruction (`dst = src0 + src1`) on the A5 backend across the supported dtype and shape matrix, including the dedicated 64-bit integer path (`Int64Binary`, full-load repeats plus a partial tail) and the valid-region-smaller-than-physical-tile path.

## Coverage points

- **Dtypes**: `float`, `int32_t`, `int16_t`, `half`, `int64_t`, `uint64_t`.
- **Shape matrix**:
  - Square (`64x64`, `32x32`) — baseline correctness.
  - Wide / tall (`1x1024`, `16x256`, `32x128`) — row-major stride and single-row reduction.
  - `src1` physical shape larger than `dst` / `src0` (`16x128` vs `16x64`) — valid-region handling when operands have different physical tiles.
  - Valid region smaller than physical tile (`vCols = 63`, `vCols = 31`, `vCols = 15`) — padding / tail-column semantics.
  - int64 / uint64 with `4x16` physical and `4x15` valid — partial valid column on the `Int64Binary` tail path.
- **Inplace variant**: `LaunchTAddInplace` reuses `dst` as `src0` to verify the aliasing path.

## Golden generation

`gen_data.py` produces NumPy reference outputs and writes them to the per-case golden directory; `main.cpp` loads them via `GetGoldenDir()` and compares with the NPU output as follows: `int64` / `uint64` results are compared exactly (`ResultCmpExact`), all other dtypes are compared with a `0.001` tolerance (`ResultCmp`).

## Out of scope

- A2/A3 backend — covered by `tests/npu/a2a3/src/st/testcase/tadd`.
- Auto-mode orchestration — covered by `tests/npu/a5/src/st/testcase/tadd_tdiv` and the auto-mode runner.
