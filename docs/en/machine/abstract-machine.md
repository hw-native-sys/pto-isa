# PTO Machine Model (Abstract)

This document defines the abstract machine model for the **PTO ISA** as exposed by **PTO Tile Lib**.

The goal is to provide a stable programming model across fast-evolving device generations:

- Different products may vary in instruction set details, on-chip storage layout, and scheduling behavior.
- PTO abstracts these differences so code can be portable, while still allowing expert users to manage memory placement, synchronization, and scheduling when needed.

## Programming styles: PTO-Auto vs PTO-Manual

PTO supports two complementary usage styles:

- **PTO-Auto**: prioritize productivity and portability.
  - The compiler/runtime selects memory placement.
  - The compiler inserts required synchronization.
  - The compiler schedules operations (and applies fusions such as VF fusion when available).
- **PTO-Manual**: prioritize control and peak performance.
  - The programmer controls memory placement and binding (e.g., `TASSIGN`).
  - The programmer is responsible for expressing ordering/synchronization (e.g., events, event synchronization).
  - The programmer controls the operation schedule; the compiler focuses on local transformations and fusions.

In practice, many applications mix both: start in PTO-Auto, then manually optimize the critical kernels.

## Execution granularity: tile programs and tile graphs

PTO programs are written at **tile granularity**:

- A **Tile** is the unit of on-chip storage and the unit of most computations.
- A **tile program** is a sequence of PTO operations over tiles, global tensors, scalars, and events.
- A **tile graph** is a collection of tile programs (or blocks) with explicit data dependencies, typically induced by global-memory reads/writes and event ordering.

The PTO machine model explains how these programs are executed on the device.

## Machine hierarchy

The model uses three conceptual layers:

- **PTO Core Machine**: the minimal execution agent that runs a single tile instruction sequence.
- **PTO Device Machine**: a collection of Core Machines plus a scheduler that maps tile blocks onto cores and enforces global-memory dependencies.
- **PTO Host Machine**: the host-side system that prepares work (compilation, caching, graph scheduling) and submits it to the device.

Additional conceptual components may exist depending on the platform:

- **Collective communication unit**: inter-device communication (multi-card / multi-node).
- **DMA / prefetch engines**: background data movement between global memory and on-chip caches.

## PTO Core Machine

A **Core Machine** executes a tile instruction stream. The model exposes the following programmer-visible concepts.

### Scalar control (Scalar Unit)

The core includes a **Scalar Unit (SU)** that drives:

- Control flow (branches, loops in generated code).
- Address calculations for memory operations.
- Event operations and explicit synchronization.

From the programming model perspective, scalar operations are “fast control” compared to tile operations, and are used to orchestrate tile compute and movement.

### Tile execution engines

The core provides multiple tile execution engines (pipelines). Exact names vary by target, but conceptually include:

- **Vector pipeline**: elementwise tile operations (add, mul, compare, etc.).
- **Matrix/cube pipeline**: matrix multiply / specialized matrix operations.
- **Memory pipeline(s)**: global-memory movement and layout transforms.
- **Fixed-function units**: target-specific accelerators (optional).

The ISA assigns each instruction to a pipeline class. Events can express ordering between pipeline classes without requiring a full-core barrier.

### Tile storage and shared memory

Each core has on-chip storage, modeled as:

- **Tile storage**: a register-file-like storage for tile objects (e.g., `Vec`, `Mat`, `Left`, `Right`, `Acc`).
- **Shared memory** (optional in the model): a core-local scratchpad for data exchange and/or cooperative patterns.

The instruction pages in `docs/en/isa/` define which tile types and layouts are legal for each instruction.

## PTO Device Machine

The **Device Machine** manages many tile blocks concurrently:

- It decomposes work into **tile blocks** (the exact unit is implementation-defined; typically a small tile program or a region of a tile graph).
- It schedules tile blocks onto available Core Machines.
- It tracks **global-memory dependencies** so that reads/writes observed through `TLOAD`/`TSTORE` occur in a legal order.

### Ordering and dependencies

The abstract rules are:

- **Within a tile block on one core**, program order is preserved for operations that have explicit data or event dependencies.
- **Across tile blocks and cores**, ordering is only guaranteed when expressed via:
  - Global-memory dependence (e.g., a `TSTORE` producing data later consumed by a `TLOAD`), as defined by the runtime/driver contract; and/or
  - Explicit events/synchronization, as defined in `docs/en/coding/Event.md` and `docs/en/coding/Event.md`.

Device implementations may execute independent tile blocks out of order and in parallel.

### MPMD scheduling (task id)

While many kernels are written in an SPMD style (all cores run the same entry function), the Device Machine model also
allows an MPMD style where different cores execute different tile programs.

In the abstract model, this is represented as the scheduler selecting a tile block (program) and mapping it to a core.
Implementations may expose this as:

- multiple kernel entry points (one per program), and/or
- a scheduler-provided **task id** passed into a single entry function, used to dispatch the program body

See also:

- `docs/en/coding/ProgrammingModel.md`

## PTO Host Machine

The **Host Machine** is responsible for preparing and submitting work to the device. Common responsibilities include:

- Compiling/JITing tile code and caching compiled functions.
- Building and optimizing tile graphs (scheduling, partitioning, replacement).
- Allocating and managing global memory buffers.
- Submitting work to one or more Device Machines and coordinating completion.

From the ISA perspective, host behavior is out of scope; it is described here only to clarify where compilation and scheduling decisions may live in an end-to-end system.
