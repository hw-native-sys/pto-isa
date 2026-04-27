# Debugging

Use this reference to shorten the path from symptom to root cause for Claude Code, Codex, or any similar coding agent working in PTO-ISA.

## Fast Triage

- Fails on CPU-SIM only:
  - inspect [include/pto/cpu](../../../../include/pto/cpu)
  - run a single testcase with `tests/run_cpu.py`
- Fails in costmodel only:
  - reproduce with [tests/run_costmodel.py](../../../../tests/run_costmodel.py)
- Fails on `sim` or `npu` only:
  - reproduce with [tests/script/run_st.py](../../../../tests/script/run_st.py)
  - compare the relevant backend under `include/pto/npu/a2a3` or `include/pto/npu/a5`
- Fails in textual assembly or lowering:
  - compare the emitted PTO-AS against [docs/isa/syntax-and-operands/assembly-model.md](../../../../docs/isa/syntax-and-operands/assembly-model.md)

## Recommended Debug Loop

1. Reproduce with the smallest testcase and `gtest_filter`.
2. Identify the exact instruction and backend.
3. Check public constraints in:
   - [include/pto/common/pto_instr.hpp](../../../../include/pto/common/pto_instr.hpp)
   - [docs/isa](../../../../docs/isa)
4. Check backend assertions and helper code.
5. Match the symptom to [review-derived-guardrails.md](review-derived-guardrails.md) if it touches pipe sync, layout/stride math, template dispatch, ST registration, docs, or hot loops.
6. Add or update a focused regression test.

## Useful Commands

### CPU-SIM

```bash
python3 tests/run_cpu.py --testcase tmatmul --gtest_filter 'TMATMULTest.*' --verbose --clean
python3 tests/run_cpu.py --testcase tpushpop --verbose --clean
```

### Costmodel

```bash
python3 tests/run_costmodel.py --testcase tcolmax --generator Ninja --clean
```

### NPU ST

```bash
python3 tests/script/run_st.py -r sim -v a3 -t tmatmul -g TMATMULTest.case1
python3 tests/script/run_st.py -r npu -v a5 -t tpushpop_cv -g TPushPopCvTest.case1_half_single_tile
```

## Typical Root Causes

- unsupported tile location or layout
- backend-only dtype or buffer resource
- missing synchronization or wrong event ordering
- A2/A3 versus A5 implementation divergence
- PTO-AS text that is legal syntactically but not accepted by the selected backend
- simulator versus hardware environment mismatch, especially `ASCEND_HOME_PATH`
- shared CPU-SIM pipe state keyed without every storage-shaping dimension
- split-lane code using compile-time lane assumptions when runtime `subblock_dim` is narrower
- layout tests whose golden data does not distinguish row-major, col-major, flattened-row, or multi-column paths

## Where to Look for Backend-Specific Behavior

- CPU-SIM:
  - [include/pto/cpu](../../../../include/pto/cpu)
- A2/A3:
  - [include/pto/npu/a2a3](../../../../include/pto/npu/a2a3)
- A5:
  - [include/pto/npu/a5](../../../../include/pto/npu/a5)
- Cross-core TPUSH/TPOP details:
  - [docs/reference/pto-cvid-cluster-id-mapping.md](../../../../docs/reference/pto-cvid-cluster-id-mapping.md)

## Debugging PTO-AS and Toolchain Interaction

- Start from the PTO-AS text and confirm it matches the documented grammar.
- Check whether the operands imply a backend-specific location or layout form.
- Compare the intended instruction family with the implementation overloads in [include/pto/common/pto_instr.hpp](../../../../include/pto/common/pto_instr.hpp).
- If the same semantic operation is expressed with multiple equivalent tile encodings, verify whether CPU-SIM and the target NPU backend normalize them the same way.

## Documentation Updates During Debugging

If you discover a real backend restriction or portability caveat:

- fix the code if it is a bug
- add or update a testcase
- document the restriction in the nearest instruction or backend reference doc
- for docs-only fixes, verify signatures and examples against headers or PTO-AS, then run strict docs/link checks when available
