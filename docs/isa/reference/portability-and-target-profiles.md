# Portability And Target Profiles

PTO is portable at the virtual-ISA level, not at the level of every target-specific optimization or support subset.

## Portable PTO Contract

Portable PTO documentation should describe:

- architecture-visible semantics of legal programs
- the required synchronization and visibility edges
- the meaning of tile, vector, scalar/control, and communication instructions

## Target Narrowing

Target profiles may narrow:

- supported data types
- supported layouts or tile roles
- supported vector forms and pipeline features
- supported performance-oriented or irregular instruction sets

When the manual records timing data, it should keep A2A3 and A5 separate. Different instructions may have different latency and throughput on Ascend 910B/910C versus Ascend 950 PR/DT, so those numbers should not be merged into one ambiguous “NPU” table.

These restrictions must be documented as target-profile restrictions, not as redefinitions of PTO itself.
