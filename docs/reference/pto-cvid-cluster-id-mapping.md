# Appendix: Cluster ID Mapping and Core Architecture Assumptions

## Overview

This appendix describes the **cluster ID (CVID) mapping** assumptions for A5 and A2A3 platforms, which underpins the TPUSH/TPOP ring buffer communication design.

## Recommended Approach: Logical Block ID as Cluster ID

When **`block_dim <= number of cores`**, the simplest and recommended approach is to use the **logical block ID directly as the cluster ID**:

```cpp
// Recommended: logical block_idx as cluster_id
int cluster_id = get_block_idx();

// GM_SLOT_BUFFER access
my_gm_slot_buffer = GM_SLOT_BUFFER_BASE + cluster_id * PER_CLUSTER_SLOT_BUFFER_SIZE;
```

### Why This Works

1. **Hardware allocates block_idx**: The FFTS block scheduler assigns `block_idx` values when launching tasks. This is a hardware-provided logical identifier.

2. **1:1 mapping**: When `block_dim <= num_cores`, each logical block maps to exactly one physical cluster — no over-subscription.

3. **No GM communication needed**: Both Cube and Vector cores can use `get_block_idx()` directly without runtime negotiation.

4. **No working buffer reservation**: The 12.5KB `cv_comm_buf` region for CVID exchange is not required.

### Kernel Identification

Kernels identify their cluster membership using hardware-provided IDs:

```cpp
// On Cube (AIC)
int my_cluster = get_block_idx();

// On Vector (AIV)
int my_cluster = get_block_idx();
int my_aiv_idx = get_subblockid();  // 0 or 1
```

## Platform Architecture Comparison

| Aspect | A5 | A2A3 |
|--------|-----|------|
| **Architecture** | Tightly coupled | Decoupled |
| **Cluster Binding** | Hardware-fixed 1:2 mapping | Task scheduler-bound |
| **Sync Mechanism** | SET intra-block | SET cross-core via FFTS |
| **Local Datapath** | L0C↔UB, UB↔L1 direct | Via GM staging |

## Cross-Core Synchronization Mechanism

### FFTS Semaphore IDs

Each cluster has **16 semaphore IDs** available for cross-core synchronization via `set_cross_core` and `wait_flag_dev`:

```
Cluster Semaphore Resources:

    +---------------------------------------------------------------+
    |  16 Semaphore IDs per Cluster (ID 0-15)                       |
    |                                                               |
    |  Each ID has a 4-bit semaphore value (0-15)                   |
    |  Can control 0-15 FIFO slots per semaphore                    |
    |                                                               |
    +---------------------------------------------------------------+
```

### TPUSH/TPOP Semaphore Allocation

TPUSH/TPOP uses **4 semaphore IDs** for bidirectional Cube-Vector communication:

| ID | Direction | Purpose |
|----|-----------|---------|
| 0 | C→V | Cube signals data ready for Vector |
| 1 | C→V | Vector signals slot free to Cube |
| 2 | V→C | Vector signals data ready for Cube |
| 3 | V→C | Cube signals slot free to Vector |

### Semaphore Operations

```cpp
// Producer signals data ready (increment semaphore by 1)
// pipe: VEC, MTE, CUBE, or FIX (avoids SU barrier)
// Uses mode2 for 1:2 cluster configuration
set_cross_core(pipe, semaphore_id);

// Consumer waits for data (decrement semaphore, blocks if 0)
wait_flag_dev(semaphore_id);
```

**Constraints**:
- Increment is always **1** (not configurable)
- Must specify **pipe** (VEC/MTE/CUBE/FIX) to avoid SU barrier stalls
- Uses **mode2** for 1:2 cluster configuration

### Mode2 Semantics (1:2 Configuration)

Under the 1:2 cluster configuration, `set_cross_core` and `wait_flag_dev` have special broadcast/reduce semantics:

| Direction | Operation | Semantics |
|-----------|-----------|-----------|
| **C→V** | `set_cross_core` | **Broadcast**: Block sets semaphore for both subblocks (AIV0 + AIV1) |
| **C→V** | `wait_flag_dev` | Each Vector core waits independently |
| **V→C** | `set_cross_core` | Each Vector core sets its own semaphore |
| **V→C** | `wait_flag_dev` | **Reduce**: Cube waits for **both** Vector subblocks to set |

```
C→V Broadcast (set_cross_core from Cube):

    AIC ──set──┬──> AIV0 semaphore++
               └──> AIV1 semaphore++

V→C Reduce (wait_flag_dev on Cube):

    AIV0 ──set──┐
                ├──> AIC waits for BOTH
    AIV1 ──set──┘
```

This ensures correct synchronization for the 1:2 Cube-Vector cluster topology without requiring separate signaling to each Vector core.

### 4-bit Semaphore Range

Each semaphore ID has a **4-bit counter** (values 0-15), which limits the maximum number of outstanding FIFO slots:

```
Semaphore value range: 0-15

    - Value 0: No slots available (consumer blocks on wait_flag_dev)
    - Value 1-15: N slots available
    - Maximum outstanding slots: 15 per direction
```

This matches the ring buffer design where each direction can have up to 8 slots (well within the 15-slot semaphore limit).

## Cluster Binding Flow

### Hardware Block/Subblock Allocation

The `block_idx` and `subblock_id` are **allocated by hardware** (FFTS block scheduler), not by software. When FFTS launches a mixed kernel, it creates logical clusters with a 1:2 block-subblock relationship:

```
FFTS Mixed Kernel Kickstart:

    +---------------------------------------------------------------------+
    |  FFTS Block Scheduler (Hardware)                                    |
    |                                                                     |
    |  Allocates: block_idx, subblock_id per core                         |
    |  Creates: 1 block + 2 subblocks (1:2 ratio) per cluster            |
    |                                                                     |
    |  +-------------------+  +-------------------+                       |
    |  | Cluster 0         |  | Cluster 1         |  ...                  |
    |  |   block_idx=0     |  |   block_idx=1     |                       |
    |  |   AIC (block)     |  |   AIC (block)     |                       |
    |  |   AIV0 (subblk 0) |  |   AIV0 (subblk 0) |                       |
    |  |   AIV1 (subblk 1) |  |   AIV1 (subblk 1) |                       |
    |  +-------------------+  +-------------------+                       |
    |                                                                     |
    +---------------------------------------------------------------------+
```

### AICPU Handshake for Core Mapping

When AICPU needs to launch a runtime on a cluster, it must **obtain the core-to-block/subblock mapping via handshake** with the hardware scheduler, rather than allocating these IDs itself:

```
AICPU Runtime Launch:

    +---------------------------------------------------------------------+
    |  AICPU                                                              |
    |                                                                     |
    |  1. Request cluster allocation from HW scheduler                    |
    |  2. Receive mapping: physical_core_id <-> (block_idx, subblock_id) |
    |  3. Initialize ffts_addr for cross-core synchronization             |
    |  4. Launch runtime on assigned cores with consistent block_idx      |
    |                                                                     |
    +---------------------------------------------------------------------+
                |
                | Handshake
                v
    +---------------------------------------------------------------------+
    |  FFTS / HW Scheduler                                                |
    |                                                                     |
    |  Provides: block_idx, subblock_id assignments for physical cores    |
    |  Ensures: same 1:2 cluster structure as kernel launches            |
    |                                                                     |
    +---------------------------------------------------------------------+
```

This ensures TPUSH/TPOP ring buffer operations work correctly whether launched via FFTS directly or through AICPU runtime.

### A3 ffts_addr Initialization

On A3, the `ffts_addr` must be initialized during the AICPU handshake process to enable cross-core synchronization via `set_cross_core` and `wait_flag_dev`:

- **ffts_addr**: Base address for FFTS semaphore registers
- **Initialization timing**: Must be done before any cross-core sync operations
- **Scope**: Per-cluster, shared by all cores (AIC + AIV0 + AIV1) in the cluster

This initialization is part of the AICPU handshake (step 3 above) and ensures the semaphore IDs (0-15) are correctly mapped to hardware registers for the assigned cluster.

## A3 FFTS Scheduler and Logical Cluster Setup

### Current TPUSH/TPOP Implementation on A3

The A3 TPUSH/TPOP implementation relies on **FFTS cross-core synchronization** features. During mixed kernel kickstart, the FFTS hardware establishes a logical cluster through the block scheduler.

### Logical-to-Physical Core Mapping

The FFTS hardware builds the logical-to-physical core mapping at task launch time:

1. **Block ID → Physical Cube Core**: The block scheduler assigns a physical AIC core to each logical block.

2. **Subblock ID → Physical Vector Core**: Each subblock (0, 1) maps to a physical AIV core that becomes a buddy of the assigned Cube.

3. **Intra-cluster sync resolution**: The FFTS hardware resolves all intra-cluster synchronization and communication paths based on this mapping.


---

## Appendix A: Generic Core ID-Based CVID Computation

> **Note**: This appendix documents the generic implementation used for `block_dim > num_cores` SIMD mode. **Not recommended for PyPTO with MPMD AICPU runtime** — use the logical block_idx approach described in the main document instead.

### Constants Reference

| Constant | A5 Value | A2A3 Value | Description |
|----------|----------|------------|-------------|
| `CORE_PER_DIE` | 18 | 25 | Clusters per die |
| `AIV_RATIO` | 2 | 2 | Vector cores per Cube |
| `AIC_AIV_PER_DIE` | 54 | 75 | Total cores per die (AIC + AIV) |
| `SEMAPHORE_IDS` | 16 | 16 | Semaphore IDs per cluster |
| `TPUSH_TPOP_SEMA_IDS` | 4 | 4 | IDs used for CV bidirectional comm |
| `SEMA_BITS` | 4 | 4 | Bits per semaphore (0-15 slots) |
| `CV_MAX_CORES` | 36 | 25 | Max clusters supported |


### Related Documents

- [Source: A5 TSyncCVID.hpp](https://gitcode.com/cann/pto-isa/blob/master/include/pto/npu/a5/custom/TSyncCVID.hpp)
- [Source: A2A3 TSyncCVID.hpp](https://gitcode.com/cann/pto-isa/blob/master/include/pto/npu/a2a3/custom/TSyncCVID.hpp)


This section documents the **generic implementation** that computes cluster ID from physical core ID. This is provided as a fallback for scenarios where **`block_dim > num_cores`** (SIMD over-subscription) or when direct block_idx mapping is not available.

### A5: Direct Core ID Computation

On A5, each die contains **18 core clusters** with a fixed 1:2 architecture. The cluster ID is computed directly from the physical core ID:

```cpp
// A5 TSYNC_CVID implementation (generic)
#ifdef __DAV_CUBE__
    int die_id = get_coreid() / AIC_AIV_PER_DIE;     // AIC_AIV_PER_DIE = 54
    comm_slot = die_id * CORE_PER_DIE + get_coreid() % AIC_AIV_PER_DIE;
#elif defined(__DAV_VEC__)
    int die_id = get_coreid() / AIC_AIV_PER_DIE;
    comm_slot = die_id * CORE_PER_DIE +
                (((get_coreid() % AIC_AIV_PER_DIE) - CORE_PER_DIE - get_subblockid()) / AIV_RATIO);
#endif
```

**Key properties**:
- **No runtime communication** needed
- **Deterministic mapping**: core ID → cluster ID is a pure function
- **Hardware-enforced 1:2 relationship**: L0C↔UB and UB↔L1 local datapaths exist within each cluster

### A2A3: Core ID via GM Exchange (Generic)

On A2A3, when using the generic implementation, the cluster ID is communicated through GM:

```cpp
// A2A3 TSYNC_CVID implementation (generic)
#ifdef __DAV_CUBE__
    // Cube core writes its core ID to GM slot
    comm_slot = static_cast<int>(get_coreid() & 0x7f);
    comm_slot %= CV_MAX_CORES;

    // Write to GM slot and flush cache
    __gm__ volatile uint32_t *comm_slot_ptr = reinterpret_cast<__gm__ volatile uint32_t *>(
        cv_comm_buf + static_cast<std::size_t>(block_idx) * CV_COMM_SLOT_BYTES);
    comm_slot_ptr[0] = static_cast<uint32_t>(comm_slot);
    dcci(comm_slot_ptr, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);

    // Signal Vector cores via FFTS
    ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(CV_CORE_SYNC, CV_COMM_CTRL));

#elif defined(__DAV_VEC__)
    // Vector core waits for Cube's signal, then reads cluster ID from GM
    __gm__ volatile uint32_t *comm_slot_ptr = reinterpret_cast<__gm__ volatile uint32_t *>(
        cv_comm_buf + static_cast<std::size_t>(block_idx) * CV_COMM_SLOT_BYTES);
    dcci(comm_slot_ptr, SINGLE_CACHE_LINE);
    wait_flag_dev(CV_COMM_CTRL);
    comm_slot = static_cast<int>(comm_slot_ptr[0]);
#endif
```

---

## Appendix B: A2A3 Working Buffer Reservation (Generic Implementation)

When using the **generic core ID-based CVID computation** (Appendix A), A2A3 requires a reserved region at the bottom of the working buffer for `cv_comm_buf` slots. This is needed for scenarios where **`block_dim > num_cores`** (SIMD over-subscription).

### Reserved Space Calculation

```
CV_COMM_SLOT_BYTES = 512 bytes (per block, 512B aligned)
CV_MAX_CORES       = 25 (max block_dim)

Reserved space = CV_COMM_SLOT_BYTES * CV_MAX_CORES
               = 512 * 25
               = 12,800 bytes
               = 12.5 KB (round up to 16KB for alignment)
```

**Note**: This reservation is **not required** when using the recommended `block_idx` as cluster ID approach (when `block_dim <= num_cores`).

### Memory Layout (Generic Implementation)

```
A2A3 Working Buffer (GM) - Generic Implementation Only:

    +------------------------------------------------------------------+
    |  Bottom 12.5KB: Reserved for cv_comm_buf (CVID negotiation)      |
    |                                                                  |
    |  +------------+------------+------------+-----+------------+     |
    |  | block_idx=0| block_idx=1| block_idx=2| ... | block_idx=24|    |
    |  |   512B     |   512B     |   512B     |     |   512B      |    |
    |  +------------+------------+------------+-----+------------+     |
    |                                                                  |
    +------------------------------------------------------------------+
    |  Remaining space: Available for GM_SLOT_BUFFER, task data, etc.  |
    +------------------------------------------------------------------+
```

### A5: No Working Buffer Reservation Needed

On A5, CVID is computed directly from `get_coreid()` without any GM communication. No working buffer reservation is required regardless of `block_dim`.

### Constants (Generic Implementation Only)

| Constant | A5 Value | A2A3 Value | Description |
|----------|----------|------------|-------------|
| `CV_COMM_SLOT_BYTES` | 512 | 512 | Bytes per block's comm slot |
| `CV_COMM_RESERVED` | 0 | 12.5KB | Working buffer reservation |
