# MSCATTER indirect-call repro (expected-fail regression test)

Standalone reproducer for the runtime-dispatch limitation documented in
[`tests/npu/a5/src/st/testcase/mgather/MGATHER.md`][mgather-doc]
§"Runtime Dispatch Requirement".

This test is **expected to fail / hang** on the current toolchain
(bisheng 15.0.5, dav-c310-vec, CANN 9.1.T500). It is **opt-in** — built
only when `TEST_CASE=mscatter_indirect_call_repro` is explicitly
supplied, so it does not slow down the default CI matrix.

When the toolchain / firmware learns to install SIMT-aware lowering for
indirect calls into SIMT-bearing functions, this test will start passing
without any test-side change — that is the regression-detector value.

[mgather-doc]: ../mgather/MGATHER.md

## What it tests

A single ELEM2D case (`float, 8 × 32 → 256-slot table, random idx`),
identical in shape, data, and golden to the corresponding case in
`tests/npu/a5/src/st/testcase/mscatter`:

```
MSCATTERTest.case_elem2d_float_8x32_random_256size  →  PASS (~5s)
MSCATTERIndirectCallReproTest
  .case_indirect_call_elem2d_float_8x32_random_256size  →  hang
```

The only difference between the two is the **call form** that reaches
the MSCATTER body:

```cpp
// In the original mscatter test (PASS):
extern "C" __global__ AICORE void runMSCATTER_NAME(...) {
    runElem2D<...>(out, src, indices);          // bisheng → HiIPUISD::CALL
}

// In this repro (hang):
extern "C" __global__ AICORE void runMSCATTER_indirect_call_NAME(...) {
    using FnT = void (*)(__gm__ float *, __gm__ float *, __gm__ int32_t *);
    volatile FnT fn = &simt_body_NAME;          // volatile → must roundtrip through memory
    fn(out, src, indices);                        // bisheng → HiIPUISD::LongCALL (BLR)
}
```

Both kernels:

- Are launched by host via standard `<<<1, nullptr, stream>>>`
  (`rtKernelLaunchWithHandleV2`).
- Carry identical `.ascend.meta.<func>` TLV records (firmware installs
  the same SIMT context at kernel entry).
- Reach the same `MSCATTER<...>(outGlobal, srcTile, idxTile)` with the
  same operands.
- Use the same input data and golden.

Empirically, byte-level diff of bisheng's `.aicore_binary` for the two
kernels shows that the short-branch `HiIPUISD::CALL` lowering emits a
pair of `0x1c..` LD-class instructions immediately before the BL that
the `HiIPUISD::LongCALL` lowering does not emit. Those two extra
instructions appear to install state the SIMT scheduler needs before
the first `cce::async_invoke` inside MSCATTER; the long-form lowering
skips them because the call target is not statically resolvable.

## How to build and run

```bash
# Build the opt-in case
bash build.sh --build --npu --a5 -- -DTEST_CASE=mscatter_indirect_call_repro

# Generate data and run
cd build_out/.../tests/npu/a5/src/st/testcase/mscatter_indirect_call_repro
python3 ../testcase/mscatter_indirect_call_repro/gen_data.py
./bin/mscatter_indirect_call_repro
# → expected: hang on aclrtSynchronizeStream (kill after ~200s)
```

## When does this start passing

Any of the following would let this test go green:

1. bisheng emits SIMT-aware setup on `HiIPUISD::LongCALL` lowering paths
   too (currently only the short-branch lowering emits it).
2. bisheng exposes a user-callable builtin (e.g.
   `__builtin_cce_simt_call_prelude(target_pc)`) so dispatcher code can
   manually install the setup before each indirect call.
3. Firmware gains a SIMT scheduler restore path triggerable from
   AICore-side intrinsics, so the runtime cost moves out of bisheng.

## Related

- Upstream description of the limitation:
  [`tests/npu/a5/src/st/testcase/mgather/MGATHER.md`][mgather-doc]
  §"Runtime Dispatch Requirement" (lines 377–403).
- Downstream blocker (uses this exact pattern):
  [hw-native-sys/simpler#970][simpler-970].

[simpler-970]: https://github.com/hw-native-sys/simpler/issues/970
