# include/pto/npu/kirin9030/

Kirin9030 series PTO instruction implementation headers.

## Overview

- Implementations are organized per instruction (or instruction family), for example: `TAdd.hpp`, `TMatmul.hpp`, `TLoad.hpp`, `TStore.hpp`
- Includes Kirin9030-specific operator patterns and utilities where applicable

## Related

- ISA semantics and examples: `docs/en/isa/`
- Kirin9030 NPU ST tests: `tests/npu/Kirin9030/src/st/`
