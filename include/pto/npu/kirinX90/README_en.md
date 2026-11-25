# include/pto/npu/kirinX90/

KirinX90 series PTO instruction implementation headers.

## Overview

- Implementations are organized per instruction (or instruction family), for example: `TAdd.hpp`, `TMatmul.hpp`, `TLoad.hpp`, `TStore.hpp`
- Includes KirinX90-specific operator patterns and utilities where applicable

## Related

- ISA semantics and examples: `docs/en/isa/`
- KirinX90 NPU ST tests: `tests/npu/KirinX90/src/st/`, share test cases with Kirin9030.
