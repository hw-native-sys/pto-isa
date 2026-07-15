# CPU_SIM
CPU_SIM is a CPU-based backend designed for execution on CPU-only systems.
It has some limitations and differences comparing to NPU backends at this moment:
- All operations are done in synchronous mode (synchronization operations has empty implementation)
- Specific memory model to mimic NPU memory (see below)
- Multithreading support is not complete (memory access in Tile objects is not synchronized across threads, so tiles are suggested not to be shared across threads)

## Enabling CPU_SIM
You may enable CPU backend (CPU_SIM) by setting `__CPU_SIM` compiler definition. In this case, programs can be built using standard CPU-targeted compiler (gcc or clang).

Note, for compatibility with NPU-based programs, some Ascend-specific functions are implemented for CPU platform in **include/pto/common/cpu_stub.hpp**. Including this file into existing program already using NPU-backend will make it compilable for CPU with only minor changes. You may not include this file, but in this you'll have to remove all functions like aclInit, aclrtSetDevice from your code or replace them with corresponding CPU-based code if needed.

## CPU_SIM memory model
Generally, all tiles memory in CPU_SIM is allocated in system memory (contrary to NPU backend where memory is divided into host and device memory, and device itself has several different memory locations). But to make CPU_SIM memory model closer to NPU, it simulates separate memory locations corresponding to NPU architecture.

CPU_SIM memory model allocates following memory locations for each thread: UB, L1, L0A, L0B, L0C. Each of this locations is basically pre-allocated array of the size corresponding to simulating NPU architecture. TASSIGN operation uses one of these arrays to assign some memory chunk from it to the tile. I.e., if TASSIGN is called for the tile with Loc==Mat and offset 10, it will assign memory starting from the L1[10] to that tile.

Currently A2A3 and A5 architectures supported, specific architecture can be chosen using pto::NPUMemoryModel::Initialize function, that should be called once for each thread (can be omitted, in this case default A2A3 architecture will be used). For more information please refer to **include/pto/cpu/NPUMemoryModel.hpp**

Also, automated memory allocation is supported. To enable it, you should enable `__PTO_AUTO__` compiler definition. This will enable lazy allocation code that will allocate memory for the tile at first attempt to get internal memory pointer (if it was not set by TASSIGN before that moment). Note that in case of auto-allocation, memory is allocated from PC system memory regardless of tile location. So it will not overlap with L1, L0A, etc.. preallocated buffers (only TASSIGN operation uses those buffers).

### To summarize:
To work with CPU_SIM you should use one of these strategies:

- **Direct memory assignment:** Every tile should have corresponding TASSIGN operation call to assign memory directly. Proper offset should be calculated manually and provided to TASSIGN operation.
- **Automatic allocation:** Enable `__PTO_AUTO__` compiler definition, in this case memory for the tiles will be allocated automatically. You may still use TASSIGN operation if needed.

**NOTE:** You should follow one of the paths described above. If you won't do any of these actions, program will crash with segfault.
