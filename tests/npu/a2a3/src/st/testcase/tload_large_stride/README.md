# TLOAD stride ST

The host harness creates deterministic inputs and checks loaded values through public `TLOAD` calls.
No external data generation is required.

- `TLoadLargeStrideTest`: 23 cases for ND/DN/NZ, static/dynamic strides, two batches,
  destination gaps, 16-bit L1 gap boundaries, 32-bit UB gap boundaries, 64-bit outer
  strides, and partial-row padding. The large cases need slightly more than 4 GiB
  of device memory, released after each case. The ordinary stride cases use `uint8_t`;
  destination-gap cases check loaded data at the intended addresses, but do not check
  that every skipped destination byte remains unchanged.
- `TLoadConversionStrideTest`: 24 b32 ND2NZ/DN2ZN cases. `float`, `int32_t` and
  `uint32_t` use element strides 32767, 32768 and 65535. Both sides of the b16
  conversion boundary and static/dynamic stride paths are checked. Each case
  allocates less than 257 KiB of input and checks both destination C0 blocks. The
  contiguous extent is 16 elements and there is one matrix: these cases test conversion
  of the within-matrix stride, not overflow of the converted extent or matrix strides.
- `TLoadVectorLengthTest`: 48 A2/A3 and KirinX90 cases for single-row ND and
  single-column DN Mat loads. ND b8/b16/b32 cases cross 65536 elements; ND b64
  cases cross 16384 and 32768 elements. DN cases cover one-block boundaries for
  b8/b16/b32/b64 and the 16384-element b64 boundary; they do not mirror all ND
  large-length cases. Static and dynamic valid lengths are covered. Tests check every loaded
  element, zero padding up to the next 32-byte boundary, and an adjacent block left unchanged.
  Inputs after the valid range contain nonzero values to detect copying them into
  destination padding. These value checks do not prove the absence of extra hardware reads.
  Eight cases offset the GM source by one element to check unaligned addresses.
- `TLoadWideStrideTest`: A5/A6 cases for byte strides below, at and above 2^40,
  plus Shape1/Shape2 hardware loops. Burst-stride cases use `2^40 - 32` (Below),
  `2^40` (At) and `2^40 + 64` (Above) bytes. The encoded maximum is `2^40 - 1`,
  so both At and Above require fallback; the exact maximum is not registered here.
  The hardware-loop cases use a `2^40 + 4096` byte outer stride. A5 also checks
  the NC1HWC0 convolution loop, but has no separate five-dimensional FRACTAL_Z
  wide-loop case even though that layout shares `TLoad5HD`.
  Virtual memory maps only accessed pages; physical allocation is a few pages,
  but the device must support reserving a virtual address span greater than 1 TiB.
  Failure to reserve/map memory fails the test before kernel execution.

Registration depends on the backend selected by CMake:

| Backend | Large stride | Conversion stride | Vector length | Wide stride | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| A2/A3 | 23 | 24 | 48 | 0 | 95 |
| A5 | 23 | 24 | 0 | 27 | 74 |
| A6 | 11 | 0 | 0 | 13 | 24 |
| KirinX90 | 23 | 24 | 48 | 0 | 95 |
| Kirin9030 | 23 | 24 | 0 | 0 | 47 |
| Kirin9050 | 11 | 0 | 0 | 0 | 11 |

These are registered case counts, not runtime pass counts. A2/A3, A5, KirinX90 and
Kirin9030 share the base L1/UB and conversion suites. A6 and Kirin9050 use only the
UB subset because their L1 readback paths differ. A5 has 13 Mat and 13 Vec wide-stride
cases plus one NC1HWC0 case; A6 has the 13 Vec cases. The single-row/single-column
length suite is registered only for the A2/A3 and KirinX90 memory backends.

For the memory-backend gap checks, Mat ND/DN/NZ cases use 65535 and 65536 blocks.
The Vec `GapMax` cases use `2^32 - 32` bytes, not the exact `UINT32_MAX`; the
`GapOverflow` cases use `2^32` bytes. The partial ND cases transfer 31 valid bytes
with source strides constructed as `Gap + 32`, so their actual gaps are `Gap + 1`.

Small and below-boundary cases remain as controls for the fast paths. Above-boundary
cases check fallbacks on backends where the supplied values exceed the instruction fields' ranges. The same
case can stay on a native path on another backend: for example, a source stride just
above 4 GiB fits the A5/A6 40-bit field. Both groups are needed to catch regressions. The 40-bit cases
are registered only for A5/A6 and require working device VMM support for the sparse span.
A VMM allocation failure does not establish whether the kernel fallback is correct.

The table covers only `tload_large_stride`. Additional regression cases in existing suites are:

- A2/A3 [`tload_gm2mat`](../../../../../a2a3/src/st/testcase/tload_gm2mat): three NZ cases with source gaps of
  65408, 65536 and 65408 blocks (32 bytes per block), plus four gap-overflow cases for
  NC1HWC0, FRACTAL_Z (two representations) and NDC1HWC0.
- Kirin9030 [`tload_mix`](../../../../../kirin9030/src/st/testcase/tload_mix), also
  shared by KirinX90: three NZ cases around the block-gap boundary, including a destination
  row gap, and three ND cases for unaligned source strides and the byte-gap boundary.
  The NZ source gaps are 65520, 65536 and 65536 blocks; the last case has a 16-block
  destination row gap. The ND cases use source byte gaps 2, 65534 and 65536, with
  byte lengths 32, 30 and 30 respectively. These are KirinX90 block/byte-gap
  boundaries; Kirin9030 uses byte strides through the register helper instead.

These existing suites use `gen_data.py`; run them separately with `-t tload_gm2mat` or
`-t tload_mix` on the corresponding backend. Their cases are not included in the table above.

The costmodel [`tload`](../../../../../../costmodel/st/testcase/tload) suite is separate
from these NPU registrations. Its DN cases use `Stride3 == 1` and `Stride4 == Shape3`
for both static and dynamic tensors. The c10 case uses 64 valid columns to match
the transfer, and c11/c12 are static DN counterparts to c09/c10; the suite has 12 cases.
Assertions check stride/shape consistency. Profiling values and accuracy tolerances
are specified per case. These costmodel checks do not measure the A5 helper's scalar
latency or execute the sparse 40-bit fallback.

The shared register helper takes the native DMA path directly for encodable burst strides;
only overflow enters the per-burst loop. On A5, compiler-proven constant burst counts of at most
8 use bounded loop unrolling; larger or runtime-dependent counts and the other backends keep
unrolling disabled. This compile-time selection adds no runtime burst-count check. The registrations above exercise
native/fallback correctness where applicable, not scalar performance. Performance measurements must
distinguish native-path range dispatch from actual per-burst fallback and account for burst count,
transfer size and synchronization. The suite's total runtime is not a measure of per-load latency.
Building without defining `_DEBUG`
removes `PTO_ASSERT` diagnostics, not the range dispatch; defining `_DEBUG=0` still enables the assertions.
There is no public template option to disable fallback. The cases in this suite use the default L2 hint; they do not
replace the separate L2-hint tests or cover all padding/data-type combinations.

The [TLOAD interface](../../../../../../../docs/en/isa/TLOAD.md) documents the distinct gap,
stride and format-conversion constraints.

Use the normal ST runner with `-t tload_large_stride`. GTest filters can select the
suites independently. Kernel compilation does not establish runtime correctness.
