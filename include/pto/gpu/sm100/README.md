# include/pto/gpu/sm100/

Architecture-specific PTO GPU specializations for **Blackwell**.

## Role

This directory is intended to hold per-SM overrides for instructions whose optimal implementation depends on:

- MMA / tensor-core generation
- shared-memory banking and swizzle rules
- async copy behavior
- warpgroup scheduling details
- register pressure / occupancy trade-offs
- architecture-specific instructions or PTX variants

## Expected contents later

- instruction-level specializations (`TLoad`, `TStore`, `TAdd`, `TMatmul`, ...)
- tuning metadata
- tile-shape policy tables
- performance notes
- microkernel registry

This folder is a scaffold for bring-up, not a live backend yet.
