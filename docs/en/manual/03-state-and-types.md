# 3. State and types

## 3.1 Scope

This chapter defines the architecture-visible state model and the type-level contracts that PTO Virtual ISA operations consume and produce.

## 3.2 Architectural state model

The architecture models the following conceptual state:

- tile values and metadata (including valid-region metadata)
- scalar values and immediate attributes
- global memory views and addresses
- synchronization/event state visible to ordering operations

Backend-internal transient state is out of scope unless it changes architectural behavior.

## 3.3 Type classes

PTO Virtual ISA type classes include:

- tile-like values (`!pto.tile<...>` class)
- memory/global views (`!pto.memref<...>` or equivalent class)
- scalar/integer/float/index classes
- event/token class for synchronization dependencies

Each instruction family MUST define accepted type classes for each operand/result position.

## 3.4 Tile legality dimensions

Tile legality is constrained by:

- element type (`dtype`)
- shape and valid-region compatibility
- location-intent role (`Mat/Left/Right/Acc/Bias/Scale` and related forms)
- layout class and alignment constraints (backend-dependent subset)

The virtual ISA defines the legality interface; concrete support sets are backend-profile-specific.

## 3.5 Valid-region semantics

Valid-region semantics are first-class:

- semantic definitions apply to indices in the declared valid domain
- values outside valid domain are unspecified unless explicitly defined
- multi-operand operations MUST define domain compatibility rules

The standard notation uses `Rv` and `Cv` for valid rows/columns.

## 3.6 Attribute contracts

Instruction attributes (for example compare mode, rounding mode, transform mode) MUST define:

- type/domain constraints
- default behavior (if any)
- interaction with semantics and legality checks
- diagnostics requirements for invalid values

## 3.7 Diagnostics requirements

Type/state verification diagnostics SHOULD include:

- operand position
- expected type class and received type class
- relevant legality dimensions (dtype/layout/location/shape)
- deterministic error identifiers for CI stability
