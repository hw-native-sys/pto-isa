<!-- Generated from `docs/isa/reference/portability-and-target-profiles.md` -->

# Portability And Target Profiles

PTO is portable at the virtual-ISA level, not at the level of every target-specific optimization or support subset.

## Portable PTO Contract

Portable PTO documentation should describe:

- architecture-visible semantics of legal programs
- the required synchronization and visibility edges
- the meaning of tile, vector, scalar/control, and communication surfaces

## Target Narrowing

Target profiles may narrow:

- supported data types
- supported layouts or tile roles
- supported vector forms and pipeline features
- supported performance-oriented or irregular families

These restrictions must be documented as target-profile restrictions, not as redefinitions of PTO itself.
