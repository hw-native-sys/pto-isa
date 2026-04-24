#include <pto/pto-inst.hpp>

#include "pto_macro_matmul.hpp"
#include "pto_macro_fa_softmax.hpp"
#include "pto_macro_fa_gu.hpp"

using namespace pto;

constexpr int CHUNKS = 4;
constexpr int ROWS = 32;
constexpr int VROWS = 16;
constexpr int COLS = 256;
constexpr int SLOTS = 4;
constexpr int CHUNK_BYTES = ROWS * COLS * sizeof(half);
constexpr int SLOT_BYTES = CHUNKS * CHUNK_BYTES;

using VTile = Tile<TileType::Vec, half, VROWS, COLS, BLayout::RowMajor, VROWS, COLS>;
using MTile = Tile<TileType::Mat, half, ROWS, COLS, BLayout::RowMajor, ROWS, COLS>;
using VGlobal = GlobalTensor<half, Shape<1, 1, 1, VROWS, COLS>, Stride<1, 1, 1, COLS, 1>>;
using MGlobal = GlobalTensor<half, Shape<1, 1, 1, ROWS, COLS>, Stride<1, 1, 1, COLS, 1>>;

AICORE inline PipeChunk chunk(int i)
{
    return i == 0 ? PipeChunk::First : (i + 1 == CHUNKS ? PipeChunk::Last : PipeChunk::Middle);
}

template <bool Chunked>
AICORE void run_vec(__gm__ half *input, __gm__ half *fifo, int iters)
{
#ifdef __DAV_VEC__
    set_mask_norm();
    set_vector_mask(-1, -1);
    VTile tile;
    TASSIGN(tile, 0u);
    VGlobal global(input + get_subblockid() * VROWS * COLS);
    TLOAD(tile, global);
    using Pipe = TPipe<0, Direction::DIR_V2C, SLOT_BYTES, SLOTS, 2, true>;
    Pipe pipe((__gm__ void *)fifo, 0u, 0u);
    for (int i = 0; i < iters; ++i) {
        for (int j = 0; j < CHUNKS; ++j) {
            if constexpr (Chunked) {
                TPUSH<Pipe, VTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile, chunk(j), j * CHUNK_BYTES);
            } else {
                TPUSH<Pipe, VTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile, PipeChunk::Single, 0);
            }
        }
    }
#endif
}

template <bool Chunked>
AICORE void run_cube(__gm__ half *output, __gm__ half *fifo, int iters)
{
#ifdef __DAV_CUBE__
    MTile tile[2];
    TASSIGN(tile[0], 0u);
    TASSIGN(tile[1], CHUNK_BYTES);
    using Pipe = TPipe<0, Direction::DIR_V2C, SLOT_BYTES, SLOTS, 2, true>;
    Pipe pipe((__gm__ void *)fifo, 0u, (uint32_t)(uint64_t)tile[0].data());
    for (int i = 0; i < iters; ++i) {
        for (int j = 0; j < CHUNKS; ++j) {
            if constexpr (Chunked) {
                TPOP<Pipe, MTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile[j & 1], chunk(j), j * CHUNK_BYTES);
            } else {
                TPOP<Pipe, MTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile[j & 1], PipeChunk::Single, 0);
            }
        }
    }
    MGlobal global(output);
    TSTORE(global, tile[(CHUNKS - 1) & 1]);
#endif
}

template <bool Chunked>
__global__ AICORE void kernel(__gm__ half *input, __gm__ half *output, __gm__ half *fifo, int iters)
{
    run_vec<Chunked>(input, fifo, iters);
    run_cube<Chunked>(output, fifo, iters);
}

extern "C" void call_kernel(void *stream, half *input, half *output, half *fifo, int mode, int iters)
{
    if (mode) {
        kernel<true><<<1, nullptr, stream>>>(input, output, fifo, iters);
    } else {
        kernel<false><<<1, nullptr, stream>>>(input, output, fifo, iters);
    }
}
