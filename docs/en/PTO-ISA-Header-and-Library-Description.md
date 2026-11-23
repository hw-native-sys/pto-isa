# Header Files and Library Files

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T05:36:21.905Z pushedAt=2026-08-29T09:05:18.489Z -->

#### API Categories

PTO stands for Parallel Tile Operation, a virtual instruction set architecture (ISA) for tile programming defined by Ascend CANN. This repository provides 124 tile instructions under this architecture, including a set of inter-NPU communication extension instructions. These instructions cover computation and transformation scenarios such as element-wise computation, reduction, broadcast, matrix multiplication and GEMV, data movement, type conversion, layout transformation, sorting, and union computation, as well as system control scenarios such as synchronization and resource configuration. They can be uniformly called by upper-layer frameworks, operator implementations, and compilation toolchains.

PTO instructions uniformly adopt abstraction at the tile block level. The naming style is: instruction category prefix + computation name, with the first letter of each word capitalized in PascalCase style. Most instructions are prefixed with `T`, indicating operations on tile objects. Irregular memory access instructions are prefixed with `M`, where `M` stands for memory, such as `MGATHER` and `MSCATTER`. The cross-core synchronization barrier is `SYNCALL`. For convenience of description, the APIs in this document are collectively referred to as PTO APIs.

**Table 1 Key instruction categories**

| Instruction Category | Description |
| --- | --- |
| Computation and movement | Tile-level core instructions prefixed with `T`, covering element-wise computation between tile and tile or tile and scalar, type conversion, selection, row/column/partial reduction, matrix multiplication and GEMV, and data movement such as Load/Store/Mov/Gather/Scatter/Extract/Insert, for example `TADD`, `TMUL`, `TMATMUL`, `TGEMV`, `TLOAD`, `TSTORE`, `TMOV`, `TGATHER`, `TSCATTER`, and `TCVT`. |
| Inter-NPU communication | Inter-NPU communication and synchronization instructions also prefixed with `T`, covering point-to-point communication, signal synchronization, and collective communication, supporting both synchronous and asynchronous forms, such as `TGET`, `TGET_ASYNC`, `TPUT`, `TPUT_ASYNC`, `TNOTIFY`, `TWAIT`, `TTEST`, `TBROADCAST`, and `TREDUCE`. |
| Irregular memory access | Irregular gather/scatter memory access instructions prefixed with `M`, such as `MGATHER` and `MSCATTER`. |
| Resource binding | `TASSIGN`, which manually binds a tile object to an implementation-defined on-chip address, corresponding to the manual placement mode. |
| Synchronization barrier | `SYNCALL`, a cross-core synchronization barrier instruction used for execution synchronization among multiple cores. |



#### Header Files and Library Files Required for API Call

The header files of the PTO APIs are located in the `<arch>-linux/include/` subdirectory under the CANN installation directory, for example `${INSTALL_DIR}/aarch64-linux/include/pto/pto-inst.hpp`. Replace `${INSTALL_DIR}` with the storage path where the CANN software is installed. Taking installation as the root user as an example, the default path is `/usr/local/Ascend/cann`. The purpose of each header file is described in the following table.

**Table 2 Header files**

| Header File | Purpose | Corresponding Library File |
| --- | --- | --- |
| `pto/pto-inst.hpp` | The sole unified public entry header file of the PTO ISA. Upper layers only need to include `pto/pto-inst.hpp` to obtain all public APIs without including sub-header files one by one. This header file aggregates the following modules by responsibility:<br>1. **Tile type system and common utilities**: used to define tile objects, memory and buffer management, constants, kernel metadata, and common utilities.<br>2. **Instruction API declarations**: used to define the unified declarations of the 124 tile instructions. Each instruction provides two forms, automatic placement and manual placement, covering APIs such as element-wise computation, reduction, broadcast, matrix multiplication and GEMV, data movement, type conversion, layout transformation, sorting, union computation, synchronization, resource configuration, and inter-NPU communication.<br>3. **Multi-backend automatic selection**: automatically selects the implementation backend based on compile-time macros. The CPU simulation backend is enabled when `__CPU_SIM` is defined, the A2/A3 CostModel backend is enabled when `__COSTMODEL` is defined, and the NPU real-device backend is enabled when `__CCE_AICORE__` is defined, with implementations divided by SoC generation such as A2/A3, A5, and Kirin.<br>4. **Inter-NPU communication instruction library**: used to define point-to-point communication, signal synchronization, and collective communication APIs between NPUs, including both synchronous and asynchronous API sets. | None. PTO-ISA is a header-only template library. Instruction implementations are expanded into the upper-layer kernel at compile time in the form of templates and inline code, without the need to separately link a `.so` file. At compile time, it depends on the BiSheng compiler `bisheng-compiler` to compile kernels containing PTO instructions into device code. At runtime, it depends on the CANN Runtime library `libruntime.so` to provide the device execution environment. |
