# pto.vdup

`pto.vdup` is part of the [Predicate And Materialization](../../predicate-and-materialization.md) instruction set.

## Summary

Duplicate scalar or vector element to all lanes.

## Mechanism

`pto.vdup` materializes scalar or selected-lane state into a vector register. The architectural result is a new vector-register value, so the operation stays in the `pto.v*` instruction set even when its input is scalar.

## Syntax

```mlir
%result = pto.vdup %input {position = "POSITION"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>
```

## Inputs

`%input` supplies the scalar or source-lane value selected by `position`.

## Expected Outputs

`%result` is the duplicated vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

`position` selects which source element or scalar position is duplicated. The
  current PTO ISA vector instructions representation models that selector as an attribute rather than a
  separate operand.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The timing sources currently used for PTO micro-instruction pages are `~/visa.txt` and `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch.
For `pto.vdup`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |
| Steady-state throughput | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |

If software scheduling or performance modeling depends on the exact cost of `pto.vdup`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate And Materialization](../../predicate-and-materialization.md)
- Previous op in instruction set: [pto.vbr](./vbr.md)
