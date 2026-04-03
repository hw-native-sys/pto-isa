# include/pto/gpu/common/

Shared building blocks for the future PTO NVIDIA GPU backend.

Expected contents over time:

- tile layout adapters
- register/shared/global memory helpers
- warp / warpgroup scheduling helpers
- async copy abstractions
- tensor-core / MMA wrapper traits
- launch-time architecture dispatch helpers
- common validation and debug utilities

This directory is intentionally created before implementation so the backend can evolve with the same structure as existing PTO backends.
