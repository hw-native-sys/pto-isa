"""Shape-specialized GEMM kernel built with PTO-DSL.

The kernel computes ``C = A * B`` on Ascend A2/A3 with:
- persistent scheduling over ``baseM x baseN`` output tiles
- N-group swizzling for L2 reuse
- L1 panel staging with double buffering
- Ping-pong L0A/L0B buffers

Tensor contract:
- A is logical and physical ND, shape ``[m, k]``, dtype fp16.
- B is logical ``[k, n]`` and passed in GM as transposed DN storage.
- C is ND, shape ``[m, n]``, dtype fp16. Accumulation is fp32 in ACC.

Default build() arguments:
  m=6144, k=6144, n=6144
  baseM=128, baseK=64, baseN=256
  stepKa=4, stepKb=4
  blockDim=24, swizzleCountN=1
"""

import importlib


def _patch_mlir_pto_quant_type():
    """Install the QuantType symbols expected by ptodsl when they are absent.

    Some MLIR/PTO Python environments do not export ``QuantType`` even though
    ptodsl imports it while constructing tensor metadata. The GEMM kernel does
    not use quantized types, so these placeholder symbols are enough for IR
    generation. Remove this shim once the active bindings always provide
    ``mlir.dialects.pto.QuantType``.
    """
    pto_dialect = importlib.import_module("mlir.dialects.pto")
    if hasattr(pto_dialect, "QuantType"):
        return

    class _QuantType:
        INT8_SYM = object()
        INT8_ASYM = object()

    pto_dialect.QuantType = _QuantType


_patch_mlir_pto_quant_type()

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s


def build(
    m=6144,
    k=6144,
    n=6144,
    baseM=128,
    baseK=64,
    baseN=256,
    stepKa=4,
    stepKb=4,
    blockDim=24,
    swizzleCountN=1,
):
    """Build and return the shape-specialized ``Gemm`` IR module.

    The emitted kernel persistently iterates over the global ``baseM x baseN``
    tile grid. ``swizzleCountN`` groups adjacent N tiles while walking M rows so
    B panels can stay hot in L2 across multiple output rows.
    """
    assert m % baseM == 0
    assert n % baseN == 0
    assert k % baseK == 0
    assert stepKa == stepKb
    assert stepKa == 4
    assert (k // baseK) % (stepKa * 2) == 0
    assert 1 <= blockDim <= (m // baseM) * (n // baseN)
    assert 1 <= swizzleCountN <= (n // baseN)

    def meta_data():
        # PTO metadata describes the pointer ABI, GM tensor views, and on-chip
        # tile buffers used by the generated Gemm symbol.
        dtype_in = pto.float16
        dtype_out = pto.float16
        dtype_acc = pto.float32
        ptr_type_in = pto.PtrType(dtype_in)
        ptr_type_out = pto.PtrType(dtype_out)
        tensor_type_in = pto.TensorType(rank=2, dtype=dtype_in)
        tensor_type_out = pto.TensorType(rank=2, dtype=dtype_out)
        tile_view_aMat = pto.SubTensorType(
            shape=[baseM, baseK * stepKa], dtype=dtype_in
        )
        tile_view_bMat = pto.SubTensorType(
            shape=[baseK * stepKb, baseN], dtype=dtype_in
        )
        tile_view_out = pto.SubTensorType(shape=[baseM, baseN], dtype=dtype_out)

        # B is loaded from GM with DN layout and staged in a matching MAT tile.
        tile_buf_bMat_cfg = pto.TileBufConfig(
            blayout="RowMajor",
            slayout="ColMajor",
            s_fractal_size=512,
        )
        tile_buf_aMat = pto.TileBufType(
            shape=[baseM, baseK * stepKa], dtype=dtype_in, memory_space="MAT"
        )
        tile_buf_bMat = pto.TileBufType(
            shape=[baseK * stepKb, baseN],
            dtype=dtype_in,
            memory_space="MAT",
            config=tile_buf_bMat_cfg,
        )
        tile_buf_aTile = pto.TileBufType(
            shape=[baseM, baseK], dtype=dtype_in, memory_space="LEFT"
        )
        tile_buf_bTile = pto.TileBufType(
            shape=[baseK, baseN], dtype=dtype_in, memory_space="RIGHT"
        )
        tile_buf_cTile = pto.TileBufType(
            shape=[baseM, baseN], dtype=dtype_acc, memory_space="ACC"
        )

        return {
            "ptr_type_in": ptr_type_in,
            "ptr_type_out": ptr_type_out,
            "tensor_type_in": tensor_type_in,
            "tensor_type_out": tensor_type_out,
            "tile_view_aMat": tile_view_aMat,
            "tile_view_bMat": tile_view_bMat,
            "tile_view_out": tile_view_out,
            "tile_buf_aMat": tile_buf_aMat,
            "tile_buf_bMat": tile_buf_bMat,
            "tile_buf_aTile": tile_buf_aTile,
            "tile_buf_bTile": tile_buf_bTile,
            "tile_buf_cTile": tile_buf_cTile,
        }

    const = s.const

    @to_ir_module(meta_data=meta_data)
    def Gemm(
        out_ptr: "ptr_type_out",
        a_ptr: "ptr_type_in",
        b_ptr: "ptr_type_in",
    ) -> None:
        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            c2 = const(2)
            cM = const(m)
            cK = const(k)
            cN = const(n)
            cBaseM = const(baseM)
            cBaseK = const(baseK)
            cBaseN = const(baseN)
            cStepKa = const(stepKa)
            cSwizzleCountN = const(swizzleCountN)
            cSwizzleCountNM1 = cSwizzleCountN - c1
            cPanelK = cBaseK * cStepKa
            bid = s.index_cast(pto.get_block_idx())
            block_num = s.index_cast(pto.get_block_num())

            m_tiles = cM // cBaseM
            n_tiles = cN // cBaseN
            output_tiles = m_tiles * n_tiles
            k_iters = cK // cBaseK
            k_panel_iters = k_iters // cStepKa

            # GM tensor views. B uses DN layout because the host passes
            # contiguous transposed B storage while the logical GEMM still sees
            # B as [K, N].
            tvA = pto.as_tensor(
                tensor_type_in,
                ptr=a_ptr,
                shape=[cM, cK],
                strides=[cK, c1],
            )
            tvB = pto.as_tensor(
                tensor_type_in,
                ptr=b_ptr,
                shape=[cK, cN],
                strides=[c1, cK],
                layout="DN",
            )
            tvC = pto.as_tensor(
                tensor_type_out,
                ptr=out_ptr,
                shape=[cM, cN],
                strides=[cN, c1],
            )

            # L1 panels are double buffered; L0A/L0B ping-pong buffers feed the
            # unrolled K steps inside one panel.
            a_l1 = [pto.alloc_tile(tile_buf_aMat), pto.alloc_tile(tile_buf_aMat)]
            b_l1 = [pto.alloc_tile(tile_buf_bMat), pto.alloc_tile(tile_buf_bMat)]
            a_l0 = [pto.alloc_tile(tile_buf_aTile), pto.alloc_tile(tile_buf_aTile)]
            b_l0 = [pto.alloc_tile(tile_buf_bTile), pto.alloc_tile(tile_buf_bTile)]
            c_l0 = pto.alloc_tile(tile_buf_cTile)

            def load_panel(l1_idx, panel_idx, m_offset, n_offset):
                # Load one A/B K panel from GM into the selected L1 slots.
                k_panel_offset = panel_idx * cPanelK
                svA = pto.slice_view(
                    tile_view_aMat,
                    source=tvA,
                    offsets=[m_offset, k_panel_offset],
                    sizes=[cBaseM, cPanelK],
                )
                svB = pto.slice_view(
                    tile_view_bMat,
                    source=tvB,
                    offsets=[k_panel_offset, n_offset],
                    sizes=[cPanelK, cBaseN],
                )
                pto.load(svA, a_l1[l1_idx])
                pto.load(svB, b_l1[l1_idx])

            def run_unrolled_k(l1_idx, panel_idx, first_panel):
                # Move one L1 panel through L0 and accumulate into c_l0.
                for inner in range(stepKa):
                    l0_idx = inner % 2
                    k_inner_offset = const(baseK * inner)
                    tile.extract(a_l1[l1_idx], c0, k_inner_offset, a_l0[l0_idx])
                    tile.extract(b_l1[l1_idx], k_inner_offset, c0, b_l0[l0_idx])
                    if first_panel and inner == 0:
                        # The first K slice initializes ACC; all remaining
                        # slices accumulate into the same C tile.
                        pto.cond(
                            s.eq(panel_idx, c0),
                            lambda: tile.matmul(a_l0[l0_idx], b_l0[l0_idx], c_l0),
                            lambda: tile.matmul_acc(c_l0, a_l0[l0_idx], b_l0[l0_idx], c_l0),
                        )
                    else:
                        tile.matmul_acc(c_l0, a_l0[l0_idx], b_l0[l0_idx], c_l0)

            for tile_id in pto.range(bid, output_tiles, block_num):
                # N-group swizzle over the base-tile grid. swizzleCountN=1
                # degenerates to row-major order over the base-tile grid.
                tile_block_loop = (n_tiles + cSwizzleCountNM1) // cSwizzleCountN
                tile_block_span = cSwizzleCountN * m_tiles
                tile_block_idx = tile_id // tile_block_span
                in_tile_block_idx = tile_id % tile_block_span
                is_last_block = tile_block_idx == (tile_block_loop - c1)
                n_col_tail = n_tiles - cSwizzleCountN * tile_block_idx
                n_col = s.select(is_last_block, n_col_tail, cSwizzleCountN)
                base_m_idx = in_tile_block_idx // n_col
                base_n_idx = tile_block_idx * cSwizzleCountN + (in_tile_block_idx % n_col)
                odd_block = (tile_block_idx % c2) == c1
                flipped_m_idx = m_tiles - base_m_idx - c1
                base_m_idx = s.select(odd_block, flipped_m_idx, base_m_idx)

                m_offset = base_m_idx * cBaseM
                n_offset = base_n_idx * cBaseN

                for panel_idx in pto.range(c0, k_panel_iters, c2):
                    next_panel_idx = panel_idx + c1
                    load_panel(0, panel_idx, m_offset, n_offset)
                    load_panel(1, next_panel_idx, m_offset, n_offset)
                    run_unrolled_k(0, panel_idx, first_panel=True)
                    run_unrolled_k(1, next_panel_idx, first_panel=False)

                svOut = pto.slice_view(
                    tile_view_out,
                    source=tvC,
                    offsets=[m_offset, n_offset],
                    sizes=[cBaseM, cBaseN],
                )
                pto.store(c_l0, svOut)

    return Gemm


if __name__ == "__main__":
    print(build())
