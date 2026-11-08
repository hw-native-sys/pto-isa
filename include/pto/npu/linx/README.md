# Linx target backend

Linx is a peer PTO NPU target alongside A2/A3, A5, A6 and Kirin. Device code
uses the Linx compiler; CPU validation uses the Linx simulator selected by
`PTO_LINX_CPU_SIM` and `__cpu_sim__`. Linx does not select an Ascend architecture.

## Public interface and implementation

Applications include `<pto/pto-inst.hpp>` and use `pto::Tile`, `pto::TileType`,
`pto::GlobalTensor` and PTO instructions. `CmpMode` and math algorithm enums
are shared with existing targets; Linx predicate encoding is mapped explicitly. `<pto/common/type.hpp>` and
`<pto/common/pto_tile.hpp>` also route to the selected target, in either include
order. Empty compiler annotation macros live in `compat.hpp`.

`pto/common/pto_instr_basic.hpp` supplies the same public wrappers to Linx and
the existing backends, preserving their dispatch and CPU tracing. Linx adapters
translate calls to the imported TileOP implementations. Tile storage and layout
traits remain target specific: a Linx tile is a compiler-managed register, not
an Ascend UB allocation. `TASSIGN` and Ascend pipe events are not implemented.

The initial common surface is:

| Instructions | NPU implementation | CPU simulator |
| --- | --- | --- |
| TLOAD, TSTORE | Linx TLSU | Strided 2-D GM transfer |
| TADD, TSUB, TMUL, TDIV, TMAX, TMIN | Linx TileOP | Imported elementwise simulation |
| TABS, TAND, TOR, TSQRT, TEXP | Linx TileOP | Imported elementwise simulation |
| TCMP | Runtime PTO mode dispatches to native predicates | Six predicates including floating EQ |
| TSEL | Five-operand PTO interface adapts native prior-destination input | Predicate selection; destination may alias a source |
| TTRANS | Rejected: retired by Linx ISA 0.58.5 | Logical transpose |
| TMATMUL | Native Linx matrix/layout constraints | Row/column-major reference multiplication |

Every instruction above returns the common `RecordEvent` marker. Wait-event
arguments produce a compile-time diagnostic until Linx event support exists.
Only the default math algorithm is accepted. TSTORE atomic/quantization
options, TMATMUL phase overloads, and other PTO instructions are not part of
this initial common surface. Internal imported functions are not a stable PTO API.

CPU-sim currently supports static valid shapes and unboxed RowMajor/ColMajor
tiles. CUBE layouts and dynamic shapes are rejected at the adapter boundary;
they are not approximated using Ascend layouts. CPU-sim validates operation
values, not Linx timing, register allocation, Shared memory, or hardware execution.

## Validation

Host numerical tests and unsupported-configuration checks (also run by CI):

```sh
python3 tests/compile/test_linx_target.py --output-dir build/linx_cpu_sim
```

Device object compilation, using a Linx compiler with PTO 0.58.7 and explicit
indexed TLSU byte-range support:

```sh
python3 tests/compile/test_linx_target.py --mode npu \
  --compiler /path/to/linx/bin/clang++ \
  --sysroot /path/to/linx/sysroot --output-dir build/linx_npu
```

`LINX_CXX` and `LINX_SYSROOT` are equivalent environment inputs. The runner uses
`--target=linx64v5-unknown-linux-musl -mlxbc -fenable-matrix -std=c++20` and
never substitutes a host compiler for device compilation. Additional flags can
be supplied through repeated `--compiler-arg=VALUE` arguments. The NPU lane
compiles vector load/arithmetic/compare/select/store, SFU/logical, and CUBE
matrix kernels to an object and checks the unsupported TTRANS diagnostic; it
does not execute device code. CI runs the CPU lane because its runners do not install Linx LLVM.

## Imported source provenance

`impl/common`, `impl/cpu_sim` and `impl/jcore` are integrated from
[LinxISA/Linx-TileOP-API](https://github.com/LinxISA/Linx-TileOP-API) at
[`ad5a2310664041be1f2fad5be1dd8e9d682f23df`](https://github.com/LinxISA/Linx-TileOP-API/commit/ad5a2310664041be1f2fad5be1dd8e9d682f23df)
(PTO TileOP API/spec 0.58.7). The imported revision header records that snapshot.

Local integration changes include qualified include paths and header guards,
PTO source headers/formatting, removal of unused architecture import branches,
`LINX_` prefixes for native entry points adapted by the public wrappers, host
architectural type traits, public comparison-enum encoding, and a CPU thread-index fallback. Public signature
and CPU correctness adaptations live outside `impl` to simplify future updates.
