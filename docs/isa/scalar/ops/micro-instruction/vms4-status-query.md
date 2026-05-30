# PTO Micro-Instruction: VMS4 Status Query (`pto.get_vms4_sr`)

This page documents the PTO micro-instruction runtime query for the `VMS4_SR` status register. The op is part of the PTO micro-instruction surface (A5 Ascend 950 profile).

## Overview

`pto.get_vms4_sr` exposes the contents of the `VMS4_SR` hardware register to scalar code. After an exhausted [`pto.vmrgsort4`](../../../vector/ops/sfu-and-dsa-ops/vmrgsort.md) merge-sort operation, `VMS4_SR` records the per-source-list executed counts; reading it lets a kernel reason about how many elements of each input list were consumed.

## Mechanism

`pto.get_vms4_sr` is a pure scalar producer. It does not move data, does not synchronize pipelines, and does not change any architectural state. It simply reads the four 16-bit fields of `VMS4_SR` and returns them as four SSA `i16` values.

The intended pattern is to issue a `pto.vmrgsort4` that may exhaust before fully consuming all inputs, then read `VMS4_SR` to discover how far each source list advanced, and use those counts to drive the next round of sort/merge work.

## `pto.get_vms4_sr`

**Syntax:** `%list0, %list1, %list2, %list3 = pto.get_vms4_sr : i16, i16, i16, i16`

**Semantics:** Read `VMS4_SR` and return the finished element counts for source lists 0, 1, 2, and 3.

### Inputs

None.

### Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%list0` | `i16` | Finished count for source list 0 |
| `%list1` | `i16` | Finished count for source list 1 |
| `%list2` | `i16` | Finished count for source list 2 |
| `%list3` | `i16` | Finished count for source list 3 |

### Register Layout

| Bits | Meaning |
|------|---------|
| `[15:0]` | finished count for source list 0 |
| `[31:16]` | finished count for source list 1 |
| `[47:32]` | finished count for source list 2 |
| `[63:48]` | finished count for source list 3 |

```c
status = VMS4_SR;
list0 = (uint16_t)(status & 0xffff);
list1 = (uint16_t)((status >> 16) & 0xffff);
list2 = (uint16_t)((status >> 32) & 0xffff);
list3 = (uint16_t)((status >> 48) & 0xffff);
```

### Constraints

- The returned values are unsigned 16-bit counts of elements consumed from each source list.
- The intended pattern is to read `VMS4_SR` after an exhausted `pto.vmrgsort4` to determine partial-progress counts.
- The op is a pure scalar producer; it has no architectural side effects.

### Examples

```mlir
// After a partial pto.vmrgsort4, read per-list executed counts
%list0, %list1, %list2, %list3 = pto.get_vms4_sr : i16, i16, i16, i16

// Use the counts to advance the next sort round
%c0_i64 = arith.extui %list0 : i16 to i64
// ... feed back into the next vmrgsort4 setup
```

## Related Operations

- 4-way merge sort: [`pto.vmrgsort`](../../../vector/ops/sfu-and-dsa-ops/vmrgsort.md)
- Block runtime queries: [BlockDim Query Operations](./block-dim-query.md)
