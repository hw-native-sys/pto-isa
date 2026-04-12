<!-- Generated from `docs/isa/programming-model/auto-vs-manual.md` -->

# Auto Vs Manual

PTO supports both Auto and Manual programming styles because they solve different problems. The ISA manual describes the shared architecture contract; this page explains how each mode delegates responsibilities between author and tooling, and which audience benefits most from each.

## Audience Decision Tree

```
Who are you?
│
├─ Compiler / toolchain developer ──► Auto mode is a contract your tool must implement
│                                      Manual mode is what your tool emits for users
│
└─ Kernel author ──► Do you need precise pipeline control?
                       │
                       ├─ YES ──► Manual mode
                       │          (explicit TASSIGN, TSYNC, set_flag/wait_flag)
                       │
                       └─ NO  ──► Auto mode
                                   (compiler/runtime manages placement and scheduling)
```

## Auto Mode

In Auto mode, the compiler or runtime infrastructure inserts `TASSIGN`, `TSYNC`, and data-movement operations automatically. The author writes only the compute payload.

### What the Author Writes

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vec_add(Tile<float, 16, 16>& c,
             const GlobalTensor<float>& ga,
             const GlobalTensor<float>& gb,
             const GlobalTensor<float>& gc) {
    Tile<float, 16, 16> a, b;
    TLOAD(a, ga);   // compiler inserts TASSIGN before TLOAD
    TLOAD(b, gb);   // compiler inserts TASSIGN before TLOAD
    TADD(c, a, b);  // compiler inserts TSYNC between TLOAD and TADD
    TSTORE(gc, c);  // compiler inserts TSYNC between TADD and TSTORE
}
```

### What the Compiler/Runtime Inserts

```
TASSIGN(a, @tile(slot))   // auto-assign tile buffer address
TSYNC()                    // sync before next producer
TLOAD(a, ga)
TSYNC()
TASSIGN(b, @tile(slot))
TSYNC()
TLOAD(b, gb)
TSYNC()
TADD(c, a, b)
TSYNC()
TSTORE(gc, c)
```

Auto mode does NOT change PTO ISA semantics. The inserted operations are standard PTO operations, not backend-specific magic.

### Constraints

- The compiler must ensure that auto-inserted operations satisfy the same legality rules as explicit operations
- Auto mode assumes the tile shape and valid region are fully determined at compile time
- `ptoas` can insert synchronization automatically via the `--enable-insert-sync` flag

## Manual Mode

In Manual mode, the author explicitly binds tile resources and manages synchronization. This gives full control over tile placement, double-buffering, and pipeline overlap.

### What the Author Writes

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vec_add_manual(Tile<float, 16, 16>& c,
                    const GlobalTensor<float>& ga,
                    const GlobalTensor<float>& gb) {
    Tile<float, 16, 16> a, b;
    TASSIGN(a, 0x1000);        // explicit tile buffer address
    TASSIGN(b, 0x2000);
    TASSIGN(c, 0x3000);
    TLOAD(a, ga);
    TLOAD(b, gb);
    TSYNC();                    // explicit synchronization
    TADD(c, a, b);
    TSYNC();
    TSTORE(gc, c);
}
```

### Double-Buffering Example

Manual mode enables double-buffering — overlapping DMA and compute on alternating tile slots:

```cpp
// Tile slot 0 and slot 1 alternate between compute and DMA
TASSIGN(tile[0], 0x1000);
TASSIGN(tile[1], 0x2000);

// Iteration i: compute on slot 0, DMA-load next tile on slot 1
TLOAD(tile[1], gm_next);       // start DMA for next iteration
set_flag(PIPE_MTE2, PIPE_V, ID0);
wait_flag(PIPE_MTE2, PIPE_V, ID0);
TADD(c, tile[0], src[0]);      // compute on current tile
TSTORE(gm_out, c);
TSYNC();
```

## Shared Contract

Both modes still share the same ISA contract:

| Aspect | Auto | Manual |
|--------|------|--------|
| PTO ISA semantics | Identical | Identical |
| Valid-region rules | Same | Same |
| Movement semantics (TLOAD/TSTORE) | Same | Same |
| Synchronization contract | Compiler-inserted | Author-controlled |
| Resource binding | Compiler-inserted | Author-controlled |
| Tile type/layout constraints | Same | Same |
| Target profile restrictions | Same | Same |

## Cases That Are Not Allowed

- Documenting Auto mode as if it made illegal programs legal
- Treating Manual mode details as global guarantees for every PTO program
- Collapsing Auto and Manual into separate ISAs instead of two ways to author PTO programs
- Relying on auto-inserted synchronization in code that requires precise pipeline ordering (use Manual instead)

## See Also

- [Execution Agents And Target Profiles](../machine-model/execution-agents.md)
- [Synchronization And Ordering](../machine-model/ordering-and-synchronization.md)
- [Portability And Target Profiles](../reference/portability-and-target-profiles.md)
- [GlobalTensor And Data Movement](./globaltensor-and-data-movement.md)
- [Tile Sync And Config](../tile/sync-and-config.md)
