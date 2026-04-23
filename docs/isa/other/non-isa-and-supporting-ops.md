# Non-ISA And Supporting Operations

Supporting operations provide convenience semantics over tile sequences, memory allocation, quantization, and random generation. Some expand to multiple core ISA operations on backends that do not implement them natively.

## Operations

| Operation | Description | Category |
|-----------|-------------|----------|
| [TALIAS](../TALIAS.md) | Create an alias view of a tile without copying data | Alias |
| [TAXPY](../TAXPY.md) | Fused multiply-add: `dst = src0 * scalar + src1` | Fused compute |
| [TCONCAT](../TCONCAT.md) | Concatenate two tiles along a specified dimension | Tile sequence |
| [pto.tdequant](./ops/non-isa-and-supporting-ops/tdequant.md) | Dequantize a tile from quantized format | Quantize |
| [TFREE](../TFREE.md) | Free a previously allocated tile or buffer | Memory |
| [THISTOGRAM](../THISTOGRAM.md) | Compute histogram of tile values | Statistics |
| [TPACK](../TPACK.md) | Pack multiple tiles into a single tile buffer | Tile sequence |
| [TPOP](../TPOP.md) | Population count of predicate mask | Predicate |
| [TPUSH](../TPUSH.md) | Push count of predicate mask | Predicate |
| [TRANDOM](../TRANDOM.md) | Fill tile with random values | Generation |

## Mechanism

### Alias (TALIAS)

Creates a new tile view that references the same underlying storage as the source tile, without copying data. The alias and source share the same UB buffer but may have different shapes, layouts, or valid regions.

### Fused Compute (TAXPY)

Fused multiply-add: `dst = src0 * scalar + src1`. This is a convenience operation that may be implemented as a single hardware instruction or expanded to `TMUL` + `TADD`.

### Tile Sequence (TCONCAT, TPACK)

`TCONCAT` concatenates two tiles along a specified axis. `TPACK` packs multiple tiles into a single buffer for storage.

### Quantization (TDEQUANT)

Convert between floating-point and quantized integer representations. In the current authored tree:

- [pto.tquant](../tile/ops/irregular-and-complex/tquant.md) remains on the tile irregular/complex path.
- [pto.tdequant](./ops/non-isa-and-supporting-ops/tdequant.md) is documented under supporting ops.

Requires scale and zero-point tensors:

$$ \mathrm{dst} = \mathrm{round}(\mathrm{src} \times \mathrm{scale} + \mathrm{zero\_point}) $$

### Memory (TFREE)

Free a previously allocated tile or buffer. The freed storage may be reused by subsequent allocations.

### Predicate (TPOP, TPUSH)

`TPOP` computes the population count (number of set bits) in a predicate mask. `TPUSH` computes the push count (number of leading zeros before the first set bit).

### Generation (TRANDOM)

Fill a tile with random values from a specified distribution.

## Constraints

- Quantization requires valid scale (non-zero) and zero-point within representable range
- `TFREE` must not be called on a tile that is still in use
- Tile concatenation requires compatible dimensions along the concatenation axis
- `TAXPY` may be expanded to separate operations on backends that do not implement it natively

## Cases That Are Not Allowed

- **MUST NOT** use `TFREE` on a tile still in use by another operation
- **MUST NOT** use invalid scale (zero) or out-of-range zero-point for quantization
- **MUST NOT** rely on `TAXPY` being a single hardware instruction on all backends

## See Also

- [Other instruction sets](../instruction-families/other-families.md) — Instruction set overview
- [Other instruction set](../instruction-surfaces/other-instructions.md) — Instruction Set description
