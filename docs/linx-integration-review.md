# Linx NPU and CPU backend integration

[中文](linx-integration-review_zh.md)

The Linx backend provides native instruction APIs through an NPU implementation and CPU-SIM. The NPU backend is a **restricted experimental integration**: admitted profiles execute on the `gfrun` functional model, while known unsupported paths produce explicit compilation errors. CPU-SIM executes the same instruction calls on the host. Functional-model results do not establish physical NPU validation.

## Dependencies and source provenance

| Component | Revision |
| --- | --- |
| Linx-TileOP-API (`linx`) | `74d7326c521aeb1b21c52602b67c55b42b3563a3` |
| SuperScalarModel / gfrun | `637bde0ec63955c4796828b297e16a057efad3cc` |
| LinxV5 Clang 15.0.4 | `b19e8c0e184cd7e3a1fa5ab57b368cbf978c3aa8` |
| PTO-spec reference | `42067d0f2890b8c013a868bd1247ebc6437b3179` |
| Host | aarch64 Linux / GCC 15.2.1 |

Twenty transitive entry dependencies plus optional `jcore/utils.hpp` make 21 imported headers. Original SHA-256 values are recorded in [upstream.json](../tests/cpu/linx/upstream.json).

The import preserves native names, namespaces, templates, defaults, return types, hardware encodings and assembly bodies. Mechanical transformations cover license comments, include paths, guards, backend selection, CPU carrier adaptations and marked admission diagnostics. The importer regenerates and checks these transformations.

CPU `native_api.hpp`, `region.hpp` and `predicate.hpp` are generated from native declarations by replacing internal emitters with host implementations. Generation removes only marked device admission checks; upstream legality checks and rejection of retired interfaces remain. Source definitions and assembly sites include helpers and macros, so their counts are not instruction coverage counts.

## Integration architecture

| Component | Responsibility |
| --- | --- |
| `include/pto/cpu/target_config.hpp`, `include/pto/pto-inst.hpp`, common type/tile headers | Select Linx CPU or device centrally, reject mixed targets and retain Ascend defaults |
| `include/pto/npu/linx/impl` | Provide the pinned native API and instruction implementations |
| `include/pto/npu/linx/experimental.hpp` | Define unsupported-path diagnostics that remain active with NDEBUG |
| `include/pto/cpu/linx` | Implement native numeric formats, Tile storage and layouts, indexed operations, matrix operations, Shared publication and PE participation on the host |
| `tests/script/sync_linx_headers.py` | Reproduce imports, insert admission checks, generate CPU declarations and verify provenance, signatures and emitter mappings |
| `tests/cpu/linx_instruction_cases.py`, `instruction_catalog.json` | Check both public entry points and associate ordinary, explicit-dimension and associated instruction names with test profiles |
| `tests/cpu/test_linx_models.py` | Compile and execute profiles, checking diagnostics, independent goldens, canaries and model completion |
| `tests/cpu/test_linx_admission.py` | Verify explicit rejection at different optimization levels, with NDEBUG and public/direct includes |
| `tests/cpu/test_linx_native.py` | Run CPU, invalid-input and target-selection checks, sanitizers and upstream fixture compilation checks |
| `tests/run_cpu.py`, `tests/script/all_cpu_tests.py` | Register Linx as a standalone suite in the CPU test workflow |

## Instruction support

| Group | CPU: all listed profiles pass | NPU model: all listed profiles pass |
| --- | ---: | ---: |
| Ordinary instruction API names | 151/151 | 109/151 |
| Explicit dimensions (`_U`) | 11/11 | 11/11 |
| Associated (`_ASS`) | 64/64 | 1/64 |
| All active instruction API names | **226/226** | **121/226** |

All 229 CPU profiles pass. The integrated NPU backend executes 122 profiles successfully and explicitly rejects 107. Of the 226 active API names, 105 are wholly or partially restricted. These counts describe the listed profiles, not exhaustive coverage of every dtype, shape, layout or overload. Rejection is separate from numerical success; API names are not distinct hardware opcodes. The [instruction support matrix](linx-tile-instruction-support.md) lists every name, restriction and profile.

Goldens are independent of CPU results. Model diagnostics, guest completion, output and canaries all affect acceptance. MX ACC and BIAS execute independently; atomics check old values and GM updates; reload inputs differ from initial data. Scalar/Predicate utilities, helpers and retired interfaces are counted separately.

## Numerical contracts and development scope

Same-type `TEXPDIF` rounds intermediate subtraction in the operation type before exponentiation. CPU `TPACK_ASS` reads the current writer's source raw words. FP16/BF16 TEXPDIF and TRSQRT each have 128 discriminating inputs per format; these and the isolated TPACK_ASS profile pass CPU, integrated NPU and pinned upstream execution.

Experimental CPU-SIM `TRSQRT` follows the pinned model's typed SQRT→DIV contract. The pinned specification uses a high-precision reciprocal square root followed by final encoding, which can produce different results. Specification alignment, broader overload/boundary coverage and physical hardware validation remain development work.

CPU validates per-Tile capacity, valid shape, ranges, indexed alignment and Shared publication/assembly. Chip-wide resources, scheduling/cache, page tables, full undefined-value tracking and hardware race ordering are outside its simulation scope. Native TPUSH/TPOP are unavailable. See the [backend contract](../include/pto/npu/linx/README.md).

Enabling a restricted NPU path requires a pinned API/compiler/model combination that passes its independent numerical and memory checks. Its diagnostic and both language versions of the support matrix can then be updated together.

## Local validation and reproduction

| Check | Result |
| --- | --- |
| Standalone regression, including NDEBUG and ASan/UBSan | 62/62 passed |
| CPU target selection | 5/5 passed |
| Restricted profiles at `-O0/-O2`, with NDEBUG, through public/direct includes | 428/428 explicit rejections, without compiler internal crashes |
| 185 upstream compilation fixtures | 91 native syntax passes, 57 explicit experimental restrictions, 37 upstream rejections |
| Header provenance | 21 imported and three generated headers passed |

Compilation fixtures measure interface acceptance, separately from numerical instruction support. Reproduction commands for instruction execution and admission checks are in the [support matrix](linx-tile-instruction-support.md#reproduction).

```bash
python3 tests/cpu/test_linx_native.py --compiler /path/to/g++-15 \
  --linx-compiler /path/to/linx-clang++ --upstream /path/to/Linx-TileOP-API \
  --survey --sanitize --release --output-dir /tmp/linx-regression
python3 tests/script/sync_linx_headers.py --upstream /path/to/Linx-TileOP-API
```

Test output directories retain source hashes, full commands and execution evidence. Pre-commit/OAT checks pass. MkDocs builds with 27 navigation warnings; these warnings prevent strict-mode success. Engineering checks and numerical instruction acceptance are reported separately.
