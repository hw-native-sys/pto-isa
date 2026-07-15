# Appendix A. Glossary

**Tile**: A 2D on-chip operand with type, shape, layout, and valid-region metadata.

**GlobalTensor**: A typed view of global memory (GM) with shape/stride metadata used by `TLOAD`/`TSTORE`.

**Valid region**: The subset of elements in a tile that are semantically defined for an operation. Often written as `[Rv, Cv]`.

**Location**: A tile storage class / intent (e.g. `Vec`, `Mat`, `Left`, `Right`, `Acc`).

**Block**: A unit of parallel work, usually identified by `block_idx`.

**Sub-block**: A subdivision within a block/core; identified by `subblockid` when applicable.

**Pipeline**: An overlapped schedule of stages (load/transform/compute/store) coordinated by synchronization.

**TSYNC**: A synchronization instruction/abstraction used to establish ordering between stage classes.
