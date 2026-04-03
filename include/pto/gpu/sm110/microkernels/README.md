# include/pto/gpu/sm110/microkernels/

Planned home for dedicated PTO GPU microkernels targeting **sm110**.

The intent is to store instruction-family-specific kernels here, for example:

- memory movement microkernels
- elementwise/vector microkernels
- reduction microkernels
- transpose / reshape helpers
- tensor-core / matmul kernels

Recommended future organization:

- `memory/`
- `elementwise/`
- `reduction/`
- `matrix/`
- `transform/`
- `sync/`

Populate this directory only after the common GPU backend contract is defined.
