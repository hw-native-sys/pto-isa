#include <limits>
#include <type_traits>
#include "pa_tiling_struct.hpp"

#ifndef __force_inline__
#define __force_inline__ inline __attribute__((always_inline))
#endif


template <uint32_t ALIGN, typename T = uint32_t>
inline __aicore__ T RoundUp(const T val)
{
    static_assert(ALIGN != 0, "align must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T align = ALIGN;
    if (val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <uint32_t DIVISOR, typename T = uint32_t>
inline __aicore__ T CeilDiv(const T dividend)
{
    static_assert(DIVISOR != 0, "divisor must not be zero");
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    T divisor = DIVISOR;
    if (dividend + divisor - 1 < dividend) {
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T>
inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (divisor == 0 || dividend + divisor - 1 < dividend) {
        return T_MAX<T>;
    }
    return (dividend + divisor - 1) / divisor;
}

constexpr Order_t ORDER_ONLY_VALUE = ONLY_VALUE;

template <typename DTypeIn, typename DTypeOut>
__aicore__ inline void conv_v(__ubuf__ DTypeOut *dst, __ubuf__ DTypeIn *src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride)
{
    if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, __bf16>::value) {
        vconv_f322bf16r((__ubuf__ __bf16 *)dst, (__ubuf__ float *)src, repeat, dstBlockStride, srcBlockStride,
            dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, half>::value) {
        vconv_f322f16((__ubuf__ half *)dst, (__ubuf__ float *)src, repeat, dstBlockStride, srcBlockStride,
            dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, half>::value && std::is_same<DTypeOut, float>::value) {
        vconv_f162f32((__ubuf__ float *)dst, (__ubuf__ half *)src, repeat, dstBlockStride, srcBlockStride,
            dstRepeatStride, srcRepeatStride);
    } else if constexpr (std::is_same<DTypeIn, __bf16>::value && std::is_same<DTypeOut, float>::value) {
        vconv_bf162f32((__ubuf__ float *)dst, (__ubuf__ __bf16 *)src, repeat, dstBlockStride, srcBlockStride,
            dstRepeatStride, srcRepeatStride);
    } else {
        static_assert(!std::is_same<DTypeIn, DTypeIn>::value, "Unsupported conv_v dtype combination.");
    }
}

template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId)
{
    uint64_t config = 1ULL | (static_cast<uint64_t>(mode) << 4) | (static_cast<uint64_t>(flagId) << 8);
    ffts_cross_core_sync(pipe, config);
}

inline __aicore__ void DdrBarrierBeforeFfts()
{
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

constexpr uint32_t PA_L1L0_BLOCK_BYTES = 32;
constexpr uint32_t PA_GM_ND2NZ_STRIDE_LIMIT = 65536;

template <typename DataType>
__aicore__ inline void pa_gm_to_l1_nd_nd(__cbuf__ DataType *l1, __gm__ DataType *gm, uint32_t nTileActual,
    uint32_t nTileCeil, uint32_t nVal, uint32_t dTileActual, uint32_t dTileCeil, uint32_t dVal)
{
    (void)nVal;
    (void)dTileCeil;
    static constexpr uint32_t BLOCK_SIZE = PA_L1L0_BLOCK_BYTES / sizeof(DataType);
    copy_gm_to_cbuf(l1, gm, 0, 1, CeilDiv<BLOCK_SIZE>(nTileActual * dTileActual), 0, 0, PAD_NONE);
}

template <typename DataType>
__aicore__ inline void pa_gm_to_l1_nd_nz(__cbuf__ DataType *l1, __gm__ DataType *gm, uint32_t nTileActual,
    uint32_t nTileCeil, uint32_t nVal, uint32_t dTileActual, uint32_t dTileCeil, uint32_t dVal)
{
    (void)nVal;
    (void)dTileCeil;
    static constexpr uint32_t BLOCK_SIZE = PA_L1L0_BLOCK_BYTES / sizeof(DataType);
    if (dVal < PA_GM_ND2NZ_STRIDE_LIMIT) {
        if constexpr (sizeof(DataType) == 4) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(l1, gm, 0, 1, nTileActual, dTileActual, 0, dVal, nTileCeil, 1, 0);
        } else {
            copy_gm_to_cbuf_multi_nd2nz_b16(l1, gm, 0, 1, nTileActual, dTileActual, 0, dVal, nTileCeil, 1, 0);
        }
    } else {
        for (uint32_t i = 0; i < nTileActual; i++) {
            if constexpr (sizeof(DataType) == 4) {
                copy_gm_to_cbuf_multi_nd2nz_b32s(l1 + i * BLOCK_SIZE, gm + i * dVal, 0, 1, 1, dTileActual, 0, 0,
                    nTileCeil, 0, 0);
            } else {
                copy_gm_to_cbuf_multi_nd2nz_b16(l1 + i * BLOCK_SIZE, gm + i * dVal, 0, 1, 1, dTileActual, 0, 0,
                    nTileCeil, 0, 0);
            }
        }
    }
}

template <typename DataType, bool IsTranspose>
__aicore__ inline void pa_l1_to_l0_a_vector(__ca__ DataType *l0, __cbuf__ DataType *l1, uint32_t mTileCeil,
    uint32_t kPartCeil, uint32_t mSrcStride, uint32_t kSrcStride, uint32_t mDstStride, uint32_t kDstStride)
{
    (void)mTileCeil;
    (void)mSrcStride;
    (void)mDstStride;
    if constexpr (IsTranspose) {
        load_cbuf_to_ca(l0, l1, 0, kPartCeil, kSrcStride, kDstStride, 0, 1, (addr_cal_mode_t)0);
    } else {
        load_cbuf_to_ca(l0, l1, 0, kPartCeil, kSrcStride, kDstStride, 0, 0, (addr_cal_mode_t)0);
    }
}

template <typename DataType, bool IsTranspose>
__aicore__ inline void pa_l1_to_l0_b_vector(__cb__ DataType *l0, __cbuf__ DataType *l1, uint32_t nTileCeil,
    uint32_t kPartCeil, uint32_t nSrcStride, uint32_t kSrcStride, uint32_t nDstStride, uint32_t kDstStride)
{
    (void)nTileCeil;
    (void)nSrcStride;
    (void)nDstStride;
    if constexpr (IsTranspose) {
        load_cbuf_to_cb(l0, l1, 0, kPartCeil, kSrcStride, kDstStride, 0, 1, (addr_cal_mode_t)0);
    } else {
        load_cbuf_to_cb(l0, l1, 0, kPartCeil, kSrcStride, kDstStride, 0, 0, (addr_cal_mode_t)0);
    }
}

__aicore__ inline void pa_l0c_to_gm_nd_fp32(__gm__ float *gm, __cc__ float *cc, uint32_t mTileActual,
    uint32_t nTileActual, uint32_t srcStride, uint32_t dstStride, uint8_t unitFlag = 0)
{
    set_nd_para((uint64_t)1);
    pipe_barrier(PIPE_FIX);
    copy_matrix_cc_to_gm(gm, cc, 0, nTileActual, mTileActual, dstStride, srcStride, unitFlag, QuantMode_t::NoQuant, 0,
        false, true);
}

struct LoadData2dTransposeParams {
    uint16_t startIndex{0};
    uint16_t repeatTimes{0};
    uint16_t srcStride{0};
    uint16_t dstGap{0};
    uint16_t dstFracGap{0};
};

// define common const value

// FFTS Flag
constexpr int32_t QK_READY = 0;
constexpr int32_t SOFTMAX_READY = 1;
constexpr int32_t UPDATE_READY = 2;
constexpr int32_t QK_READY_DECODER = 3;
constexpr int32_t SOFTMAX_READY_DECODER = 4;
constexpr int32_t UPDATE_READY_DECODER = 5;
constexpr int32_t QK_READY_STAGE2 = 6;
constexpr int32_t SOFTMAX_READY_STAGE2 = 7;
constexpr int32_t UPDATE_READY_STAGE2 = 8;
constexpr uint32_t VEC_DEQ_K0_READY = 9;
constexpr uint32_t VEC_DEQ_K1_READY = 10;
constexpr uint32_t VEC_DEQ_V0_READY = 11;
constexpr uint32_t VEC_DEQ_V1_READY = 12;
constexpr int32_t REDUCE_READY_DECODER = 13;


constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr int64_t TMP_SIZE = 65536;              // 256 * 256
constexpr int32_t BIT_SHIFT = 8;

const int32_t TILING_BATCH = 0;
const int32_t TILING_NUMHEADS = 1;
const int32_t TILING_HEADDIM = 2;
const int32_t TILING_NUMBLOKS = 3;
const int32_t TILING_BLOCKSIZE = 4;
const int32_t TILING_MAXBLOCKS = 5;
const int32_t TILING_TOR = 6;
const int32_t TILING_KVHEADS = 7;
const int32_t TILING_FORMER_BATCH = 8;
const int32_t TILING_FORMER_HEAD = 9;
const int32_t TILING_TAIL_BATCH = 10;
const int32_t TILING_TAIL_HEAD = 11;
const int32_t TILING_HEADNUM_MOVE = 12;
const int32_t TILING_MASK_MAX_LEN = 13;
const int32_t TILING_BATCH_STRIDE = 14;
const int32_t TILING_HEAD_STRIDE = 15;
const int32_t TILING_KEY = 16;
const int32_t TILING_HEADSIZE = 17;
const int32_t TILING_PARASIZE = 18;
const int32_t TILING_GROUPNUM = 19;
const int32_t TILING_FORMER_GROUP_MOVE = 20;
const int32_t TILING_TAIL_GROUP_MOVE = 21;
const int32_t TILING_MAX_KVSEQLEN = 22;
const int32_t TILING_KVSPLIT = 23;
const int32_t TILING_KVCORENUM = 24;
const int32_t TILING_BLOCKSIZE_CALC = 25;
const int32_t TILING_TOTAL_BLOCK_NUM = 26;
const int32_t TILING_PREFILL_BS = 27;
const int32_t TILING_DECODER_BS = 28;
const int32_t TILING_HEADDIM_V = 29;
const int32_t TILING_MODCOEF = 30;
const int32_t TILING_DIVCOEF = 31;
const int32_t TILING_QHEADORIGINAL = 32;
const int32_t TILING_COMPRESSHEAD = 33;
const int32_t TILING_QUANTYPE = 34;
const int32_t TILING_DATA_SHAPE_TYPE = 35;
const int32_t TILING_SCALETYPE = 36;
const int32_t TILING_MASK_TYPE_ND = 37;
const int32_t TILING_HEADDIM_K_SPLIT = 38;
const int32_t TILING_HEADDIM_V_SPLIT = 39;
const int32_t TILING_HEADDIM_V_SPLIT_VECTOR_FORMER = 40;
const int32_t TILING_HEADDIM_V_SPLIT_VECTOR_TAIL = 41;
const int32_t BLOCKSIZE_CALC_256 = 256;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t KV_SEQ_STEP = 16;
constexpr uint32_t MAX_NUMEL_INST_B8 = 255 * 256;
constexpr uint32_t MAX_NUMEL_INST_B16 = 255 * 128;
constexpr uint32_t MAX_NUMEL_INST_B32 = 255 * 64;

using TilingKeyType = AtbOps::TilingKeyType;

using DataShapeType = AtbOps::DataShapeType;

using CompressType = AtbOps::CompressType;

using PagedAttnVariant = AtbOps::PagedAttnVariant;

template<TilingKeyType tilingKeyType>
struct AttentionType
{
};


template<>
struct AttentionType<TilingKeyType::TILING_HALF_DATA>
{
    using mm1OutputType = float;
    using mm1CopyType = float;
    using mmBiasType = float;
    using mmScaleType = float;
    using mm2OutputType = float;
    using mm2CopyType = float;
};

template<>
struct AttentionType<TilingKeyType::TILING_BF16_DATA>
{
    using mm1OutputType = float;
    using mm1CopyType = float;
    using mmBiasType = float;
    using mmScaleType = float;
    using mm2OutputType = float;
    using mm2CopyType = float;
};


#ifdef __DAV_C220_CUBE__
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384;    // 128 * 128 = 16K
constexpr int32_t L0AB_UINT8_BUF_SIZE = 16384 * 2;
constexpr int32_t L0C_FLOAT_BUF_SIZE = 16384;
constexpr int32_t L0C_UINT8_BUF_SIZE = 131072;
constexpr int32_t CUBE_MATRIX_SIZE = 256;        // 16 * 16
constexpr int64_t L0AB_UINT8_BLOCK_SIZE = 32768; // 128 * 128 * 2B
constexpr int32_t L1_HALF_BUF_SIZE = 65536;  // 256 * 256
constexpr int32_t L1_P_UINT8_BUF_SIZE = 32768;

constexpr int32_t TMP_SIZE_DECODER = 32768;

constexpr int32_t L1_HALF_BUF_SIZE_DECODER = 16384;
constexpr int32_t L1_UINT8_BUF_SIZE_DECODER = 16384 * 2;
constexpr int32_t L1_KV_HALF_BUF_SIZE = 65536;// 2* 128 * 256
constexpr int32_t L1_KV_UINT8_BUF_SIZE = 65536 * 2;
constexpr uint64_t L1_E_UINT8_SIZE = 1024;  // 32 * 32 * 1B
constexpr uint64_t L1_SCALE_UINT8_SIZE = 4096;  // uint64 256 * 8 * 2head
constexpr uint64_t L1_SCALE_UINT64_SIZE = L1_SCALE_UINT8_SIZE / 8;
constexpr uint64_t L1_OFFSET_UINT8_SIZE = 2048;  // int32 256 * 4 8 2head
constexpr uint64_t L1_OFFSET_INT32_SIZE = L1_OFFSET_UINT8_SIZE / 4;

//DeQuant
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN = 32768; // 32 KB
constexpr uint32_t L0C_PINGPONG_BUFFER_LEN_INT32 = 16384; // 65536 / 4
constexpr uint32_t CUBE_MATRIX_SIZE_512 = 16 * 32;       // 16 * 23
constexpr int32_t BLOCK_SIZE_16 = 16;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_64 = 64;
constexpr uint64_t CONST_128 = 128;
constexpr uint32_t EMBED_SPLIT = 256;
constexpr uint32_t ROUND_EMBED_SPLIT = 256;

#elif __DAV_C220_VEC__
constexpr uint32_t HALF_VECTOR_SIZE = 128;
constexpr uint32_t UB_ALIGN_BYTE = 32;
constexpr int32_t FLOAT_VECTOR_SIZE = 64;
constexpr int64_t UB_UINT8_BLOCK_SIZE_MLA = 16384;      // 96 * 128 * 2B // prefill/decoder diff
constexpr int64_t UB_UINT8_BLOCK_SIZE_NORM = 24576;
constexpr int64_t UB_UINT8_LINE_SIZE = 512;         // 64 * 4 B; 2x headroom to avoid UB overlap.
constexpr int64_t UB_HALF_LINE_SIZE = 256;          // UB_FLOAT_LINE_SIZE * 2
constexpr int64_t UB_FLOAT_LINE_SIZE = 128;         // 64 floats; 2x headroom to avoid UB overlap.

constexpr int64_t PRE_UB_UINT8_BLOCK_SIZE = 16384;  // 64 * 128 * 2B
constexpr int32_t VECTOR_SIZE = 128;                // prefill
constexpr int32_t FLOAT_BLOCK_SIZE = 8;
constexpr int32_t UB_HALF_BUF_SIZE = 8192;          // 64 * 128
constexpr int32_t TMP_SIZE_DECODER = 32768;
constexpr int32_t STAGE2_UB_UINT8_BLOCK_SIZE = 8192;
constexpr int32_t CUBE_MATRIX_SIZE = 256;
constexpr uint32_t MAX_UB_SIZE = 196608; // 192 * 1024
constexpr uint32_t EMBED_SPLIT_SM = 128;
constexpr uint32_t ROUND_EMBED_SPLIT_SM = 128;

__aicore__ __attribute__((always_inline)) void inline __set_mask(int32_t len)
{
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % FLOAT_VECTOR_SIZE;
    for (int64_t i = 0; i < temp; i++) {
        mask |= one << i;
    }

    if (len == VECTOR_SIZE) {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    } else if (len >= FLOAT_VECTOR_SIZE) {
        set_vector_mask(mask, (uint64_t)-1);
    } else {
        set_vector_mask(0x0, mask);
    }
}

template<PagedAttnVariant pagedAttnVariant>
struct UbufAlloc
{
};


template<>
struct UbufAlloc<PagedAttnVariant::DEFAULT>
{
    const uint32_t ls32_ubuf_offset = 0;
    const uint32_t lp_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_NORM;
    const uint32_t lp32_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_NORM;
    const uint32_t mask_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_NORM;
    const uint32_t lo_ubuf_offset = 3 * UB_UINT8_BLOCK_SIZE_NORM;
    const uint32_t mask32_ubuf_offset = 3 * UB_UINT8_BLOCK_SIZE_NORM;
    const uint32_t ls16_ubuf_offset = 3 * UB_UINT8_BLOCK_SIZE_NORM;
    const uint32_t lm32_ubuf_offset = 5 * UB_UINT8_BLOCK_SIZE_NORM;
    const uint32_t hm32_ubuf_offset = 5 * UB_UINT8_BLOCK_SIZE_NORM + 1 * UB_UINT8_LINE_SIZE;
    const uint32_t pm32_ubuf_offset = 5 * UB_UINT8_BLOCK_SIZE_NORM + 2 * UB_UINT8_LINE_SIZE;
    const uint32_t pm32_ubuf_stage2_offset = 5 * UB_UINT8_BLOCK_SIZE_NORM + 3 * UB_UINT8_LINE_SIZE;
    const uint32_t descale1_offset = 5 * UB_UINT8_BLOCK_SIZE_NORM + 4 * UB_UINT8_LINE_SIZE;
    const uint32_t descale2_offset = 5 * UB_UINT8_BLOCK_SIZE_NORM + 5 * UB_UINT8_LINE_SIZE;
    const uint32_t dm32_ubuf_offset = 5 * UB_UINT8_BLOCK_SIZE_NORM + 6 * UB_UINT8_LINE_SIZE;
    const uint32_t dm32_ubuf_stage2_offset = 5 * UB_UINT8_BLOCK_SIZE_NORM + 7 * UB_UINT8_LINE_SIZE;
    const uint32_t ll_ubuf_offset = MAX_UB_SIZE - (UB_UINT8_LINE_SIZE + UB_UINT8_LINE_SIZE * 4); // 2 * UB_UINT8_LINE_SIZE
    const uint32_t ll_ubuf_stage2_offset = MAX_UB_SIZE- UB_UINT8_LINE_SIZE * 2;      // 2 * UB_UINT8_LINE_SIZE
    const uint32_t gm32_ubuf_offset = dm32_ubuf_stage2_offset + 3 * UB_UINT8_LINE_SIZE; // 2 * UB_UINT8_LINE_SIZE
    const uint32_t gl_ubuf_offset = gm32_ubuf_offset + 2 * UB_UINT8_LINE_SIZE;          // 3 * UB_UINT8_LINE_SIZE
    const uint32_t gl32_ubuf_offset = gm32_ubuf_offset + 2 * UB_UINT8_LINE_SIZE;        // 3 * UB_UINT8_LINE_SIZE
    const uint32_t go_ubuf_offset = gl_ubuf_offset + 3 * UB_UINT8_LINE_SIZE;            // 16K
    const uint32_t go32_ubuf_offset = gl_ubuf_offset + 3 * UB_UINT8_LINE_SIZE;          // 16K
    const uint32_t tv32_ubuf_offset = go32_ubuf_offset + 2 * UB_UINT8_BLOCK_SIZE_NORM;
};
#endif

#ifdef __DAV_C220_CUBE__
template <bool SplitKV = false, TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA, typename IN_DTYPE = half,  typename OUT_DTYPE = half, typename IN_KVDTYPE = half, PagedAttnVariant pagedAttnVariant = PagedAttnVariant::DEFAULT, DataShapeType dataShapeType = DataShapeType::BSND, CompressType compressType = CompressType::COMPRESS_TYPE_UNDEFINED, bool SplitBlock = false>
class UnpadAttentionDecoderAic {
    // define dtype
    using mm1OutputType = typename AttentionType<tilingKeyType>::mm1OutputType;
    using mm1CopyType = typename AttentionType<tilingKeyType>::mm1CopyType;
    using mmBiasType = typename AttentionType<tilingKeyType>::mmBiasType;
    using mmScaleType = typename AttentionType<tilingKeyType>::mmScaleType;
    using mm2OutputType = typename AttentionType<tilingKeyType>::mm2OutputType;
    using mm2CopyType = typename AttentionType<tilingKeyType>::mm2CopyType;
    static constexpr uint32_t T_CUBE_MATRIX_SIZE = CUBE_MATRIX_SIZE_512 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_SIZE =  BLOCK_SIZE_32 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_OFFSET = 2 / sizeof(IN_DTYPE);

public:
    __aicore__ __attribute__((always_inline)) inline UnpadAttentionDecoderAic(uint32_t prefill_batch_size, uint32_t decoder_batch_size) {
        prefill_batch_size_ = prefill_batch_size;
        decoder_batch_size_ = decoder_batch_size;
    }

    __aicore__ __attribute__((always_inline)) inline void SetArgs(
        __gm__ uint8_t *__restrict__ sync,
        __gm__ uint8_t *__restrict__ q_in_gm,
        __gm__ uint8_t *__restrict__ k_in_gm,
        __gm__ uint8_t *__restrict__ v_in_gm,
        __gm__ uint8_t *__restrict__ block_tables_in_gm,
        __gm__ uint8_t *__restrict__ o_out_gm,
        __gm__ uint8_t *__restrict__ s_out_gm,
        __gm__ uint8_t *__restrict__ p_out_gm,
        __gm__ uint8_t *__restrict__ o_temp_gm,
        __gm__ uint8_t* __restrict__ gm_k16,
        __gm__ uint8_t* __restrict__ gm_v16,
        __gm__ uint8_t *__restrict__ tiling_para_gm,
        __gm__ uint8_t *__restrict__ razorOffset,
        uint32_t pto_block_idx,
        uint32_t pto_block_num)
    {
        if (sync != nullptr) {
            set_ffts_base_addr((uint64_t)sync);
        }
        set_padding(0);
        set_atomic_none();
        set_nd_para(1ULL);
        set_mask_norm();

        q_gm = reinterpret_cast<__gm__ IN_DTYPE *>(q_in_gm);
        k_gm = reinterpret_cast<__gm__ IN_KVDTYPE *>(k_in_gm);
        v_gm = reinterpret_cast<__gm__ IN_KVDTYPE *>(v_in_gm);
        block_tables_gm = reinterpret_cast<__gm__ int32_t *>(block_tables_in_gm);
        s_gm = reinterpret_cast<__gm__ mm1CopyType *>(s_out_gm);

        p_gm = reinterpret_cast<__gm__ IN_DTYPE *>(p_out_gm);
        o_tmp_gm = reinterpret_cast<__gm__ mm2CopyType *>(o_temp_gm);
        tiling_gm = reinterpret_cast<__gm__ uint8_t *>(tiling_para_gm);

        num_tokens = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm));
        q_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_NUMHEADS));
        embedding_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADDIM));
        embedding_size_v = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADDIM_V));
        block_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE));
        max_num_blocks_per_query = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MAXBLOCKS));
        kv_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_KVHEADS));
        former_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_FORMER_BATCH));
        former_head_split = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_FORMER_HEAD));
        tail_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_TAIL_BATCH));
        tail_head_split = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_TAIL_HEAD));
        head_split_num = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADNUM_MOVE));
        tiling_head_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADSIZE));
        tiling_para_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_PARASIZE));
        group_num = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_GROUPNUM));
        block_size_calc = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE_CALC));
        q_head_original = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_QHEADORIGINAL));
        compressHead = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_COMPRESSHEAD));
        block_idx = pto_block_idx;
        block_num = pto_block_num;
        block_size_inner_count = block_size / block_size_calc;

            former_group_num_move = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_FORMER_GROUP_MOVE));
            tail_group_num_move = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_TAIL_GROUP_MOVE));
        kv_split_per_core = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_KVSPLIT));
        kv_split_core_num = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_KVCORENUM));

        former_head_split_num = (former_head_split > group_num) && (former_group_num_move == group_num) ? head_split_num : 1;
        tail_head_split_num = (tail_head_split > group_num) && (tail_group_num_move == group_num) ? head_split_num : 1;

        stride_kv = static_cast<uint64_t>(kv_heads) * embedding_size;


        __k = embedding_size;
        round_k = RoundUp<T_BLOCK_SIZE>(__k);
    }


    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
        set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID3);
        set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID4);
        set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID5);
        set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
        core_per_batch = (q_heads + former_head_split - 1) / former_head_split;
        process_num = static_cast<uint64_t>(former_batch) * core_per_batch * kv_split_core_num;

        for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {  // for task
            uint32_t cur_batch = process / (core_per_batch * kv_split_core_num) + prefill_batch_size_;
            uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
            cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 13 + offset_tiling));
            offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
            uint32_t batch_idx = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 8 + offset_tiling));
            uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm  + 1 + offset_tiling));
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;
            uint32_t cur_head = (process / kv_split_core_num) % core_per_batch;
            uint32_t cur_nIndx = process % kv_split_core_num;
            uint32_t start_head = cur_head * former_head_split;
            uint32_t start_kv = cur_nIndx * kv_split_per_core;
            uint32_t cur_kv_seqlen = kv_split_per_core;
            uint32_t kv_loop = (kv_seqlen_align + kv_split_per_core - 1) /  kv_split_per_core;
            if (cur_nIndx >= kv_loop) {
                continue;
            }
            if (cur_nIndx == (kv_loop - 1)) {
                cur_kv_seqlen = kv_seqlen - cur_nIndx * kv_split_per_core;
            }
            uint32_t cur_head_num = former_head_split;
            if (cur_head == (core_per_batch - 1)) {
                cur_head_num = q_heads - cur_head * former_head_split;
                former_group_num_move = former_group_num_move <= cur_head_num ? former_group_num_move : cur_head_num;
            }
            uint32_t head_split_loop = (cur_head_num + (former_head_split_num * former_group_num_move) - 1) /
                                       (former_head_split_num * former_group_num_move);
                InnerRunCube(batch_idx, start_head, cur_head_num, head_split_loop, start_kv, cur_kv_seqlen, offset_tiling, former_group_num_move, former_head_split_num);
        }
        if (tail_batch > 0) {
            core_per_batch = (q_heads + tail_head_split - 1) / tail_head_split;
            process_num = static_cast<uint64_t>(tail_batch) * core_per_batch;
            for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {  // for task
                uint32_t cur_batch = process / core_per_batch + former_batch + prefill_batch_size_;
                uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
                cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 13 + offset_tiling));
                offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
                uint32_t batch_idx = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 8 + offset_tiling));
                uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
                if (kv_seqlen == 0) {
                    continue;
                }
                uint32_t cur_kv_seqlen = kv_seqlen;
                uint32_t start_kv = 0;
                uint32_t cur_head = process % core_per_batch;
                uint32_t cur_head_num = tail_head_split;
                if (cur_head == (core_per_batch - 1)) {
                    cur_head_num = q_heads - cur_head * tail_head_split;
                    tail_group_num_move = tail_group_num_move <= cur_head_num ? tail_group_num_move : cur_head_num;
                }
                uint32_t head_split_loop = (cur_head_num + (tail_head_split_num * tail_group_num_move) - 1) /
                                           (tail_head_split_num * tail_group_num_move);
                uint32_t start_head = (process % core_per_batch) * tail_head_split;
                    InnerRunCube(batch_idx, start_head, cur_head_num, head_split_loop, start_kv, cur_kv_seqlen, offset_tiling, tail_group_num_move, tail_head_split_num);
            }
        }
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID4);
        wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID5);
        wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }
private:



    __attribute__((always_inline)) inline __aicore__ void LoadQToL1(
        uint32_t q_offset,
        uint32_t cur_head_num)
    {
        if (is_multi_head_mmad) {
            // gm_to_l1q
            pa_gm_to_l1_nd_nz<IN_DTYPE>(
                l1q_buf_addr_tensor,
                q_gm + (q_offset),
                cur_head_num,        // nValue
                RoundUp<16>(cur_head_num),// dstNzC0Stride
                0,                     // dstNzMatrixStride, unused
                __k,                   // dValue
                0,                     // dstNzMatrixStride, unused
                __k                   // srcDValue
            );
        } else {
            if (embedding_size % T_BLOCK_SIZE == 0) {
                pa_gm_to_l1_nd_nd<IN_DTYPE>(
                    l1q_buf_addr_tensor,
                    q_gm + (q_offset),
                    1,
                    0,
                    0,
                    round_k * cur_head_num,               // lenBurst
                    0,
                    0
                );
            } else {
                for (uint32_t copy_idx = 0; copy_idx < cur_head_num; copy_idx++) {
                    pa_gm_to_l1_nd_nd<IN_DTYPE>(
                        l1q_buf_addr_tensor + (copy_idx * round_k),
                        q_gm + (q_offset + copy_idx * embedding_size),
                        1,
                        0,
                        0,
                        round_k,               // lenBurst
                        0,
                        0
                    );
                }
            }
        }
    }





    __attribute__((always_inline)) inline __aicore__ void LoadKVToL1(
        __gm__ IN_KVDTYPE *__restrict__ kv_gm_tensor,
        __cbuf__ IN_KVDTYPE *__restrict__ l1kv_buf_addr_tensor,
        bool move_l1b_flag,
        uint32_t head_num_move,
        uint32_t cur_batch,
        uint32_t cur_kv_seqlen,
        uint32_t start_kv,
        uint32_t qk_round_n,
        uint32_t real_n_loop,
        uint32_t sub_n_loop,
        uint32_t n_idx
    )
    {
        for (uint32_t inner_n_idx = 0; inner_n_idx < sub_n_loop; inner_n_idx++) {
            uint32_t actual_idx = n_idx * sub_n_loop + inner_n_idx;
            uint32_t sub_qk_n = block_size;
            if (actual_idx >= real_n_loop) {
                break;
            }
            uint32_t block_table_id = (uint32_t)(*(block_tables_gm +
                            cur_batch * max_num_blocks_per_query + start_kv / block_size + actual_idx));
            int64_t kv_offset = (int64_t)block_table_id * block_size * stride_kv;


            if (actual_idx == (real_n_loop - 1)) {
                sub_qk_n = (cur_kv_seqlen - actual_idx * block_size);
            }
            if (group_num == 1) {
                if (inner_n_idx == 0) {
                    wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<::event_t>(l1b_pingpong_flag + 2));
                }
                    pa_gm_to_l1_nd_nz<IN_KVDTYPE>(
                        l1kv_buf_addr_tensor + (l1b_offset + block_size * T_BLOCK_SIZE * inner_n_idx), kv_gm_tensor + (kv_offset),
                        sub_qk_n,            // nValue
                        qk_round_n,          // dstNzC0Stride
                        0,                   // dstNzMatrixStride, unused
                        __k * head_num_move, // dValue
                        0,                   // dstNzMatrixStride, unused
                        stride_kv            // srcDValue
                    );
                if (actual_idx == real_n_loop - 1 || inner_n_idx == sub_n_loop - 1) {
                    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1b_pingpong_flag + 2));
                    move_l1b_offset = l1b_offset;
                }
            } else {
                if (move_l1b_flag && inner_n_idx == 0) {
                    l1b_pingpong_flag = 1 - l1b_pingpong_flag;
                    l1b_offset = l1b_pingpong_flag * L1_KV_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
                    wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<::event_t>(l1b_pingpong_flag + 2));
                }
                if (move_l1b_flag) {
                        pa_gm_to_l1_nd_nz<IN_KVDTYPE>(
                            l1kv_buf_addr_tensor + (l1b_offset + block_size * T_BLOCK_SIZE * inner_n_idx), kv_gm_tensor + (kv_offset),
                            sub_qk_n,            // nValue
                            qk_round_n,          // dstNzC0Stride
                            0,                   // dstNzMatrixStride, unused
                            __k * head_num_move, // dValue
                            0,                   // dstNzMatrixStride, unused
                            stride_kv            // srcDValue
                        );

                    if (actual_idx == real_n_loop - 1 || inner_n_idx == sub_n_loop - 1) {
                        set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1b_pingpong_flag + 2));
                    }
                }
                move_l1b_offset = l1b_offset;
            }
        }
    }

    // antiquant
    __attribute__((always_inline)) inline __aicore__ void
    LoadKVToL1(__gm__ IN_DTYPE *__restrict__ kv_gm_tensor,        // [seq_len, num_head, embd_size]
               __cbuf__ IN_DTYPE *__restrict__ l1kv_buf_addr_tensor, // [seq_len, hidden_size]
               bool move_l1b_flag, uint32_t head_num_move, uint32_t qk_n, uint32_t qk_round_n, uint32_t num_head)
    {
        if (group_num == 1) {
            // [qk_n, cur_head_num, head_size] -> [qk_n, head_num_move, head_size]
            wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<::event_t>(l1b_pingpong_flag + 2));
            pa_gm_to_l1_nd_nz<IN_DTYPE>(l1kv_buf_addr_tensor + (l1b_offset),
                                                                                      kv_gm_tensor,
                                                                                      qk_n,       // nValue
                                                                                      qk_round_n, // dstNzC0Stride
                                                                                      0,          // dstNzMatrixStride
                                                                                      __k * head_num_move, // dValue
                                                                                      0,        // dstNzMatrixStride
                                                                                      stride_kv // srcDValue
            );
            set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1b_pingpong_flag + 2));
            move_l1b_offset = l1b_offset;
        } else {
            if (move_l1b_flag) {
                l1b_pingpong_flag = 1 - l1b_pingpong_flag;
                l1b_offset = l1b_pingpong_flag * L1_KV_HALF_BUF_SIZE;
                wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<::event_t>(l1b_pingpong_flag + 2));
                pa_gm_to_l1_nd_nz<IN_DTYPE>(l1kv_buf_addr_tensor + (l1b_offset),
                                                                                          kv_gm_tensor,
                                                                                          qk_n,       // nValue
                                                                                          qk_round_n, // dstNzC0Stride
                                                                                          0, // dstNzMatrixStride
                                                                                          __k * head_num_move, // dValue
                                                                                          0,        // dstNzMatrixStride
                                                                                          stride_kv // srcDValue

                );
                set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1b_pingpong_flag + 2));
                move_l1b_offset = l1b_offset;
            }
        }
    }


    __attribute__((always_inline)) inline __aicore__ void ProcessQK(
        __gm__ mm1CopyType *__restrict__ s_gm_tensor,
        uint32_t qk_n, uint32_t qk_round_n,
        uint32_t head_num_move, uint32_t group_num_move,
        uint32_t head_split_num_move, uint32_t cur_head_num_round,
        uint32_t split_idx, bool move_l1b_flag, bool is_l0b_pingpong_off)
    {
        uint32_t loop_mad = (group_num == 1) ? head_num_move : 1;
        for (uint32_t headdim_idx = 0; headdim_idx < loop_mad; headdim_idx++) {
            bool move_l0b_flag = move_l1b_flag;
            wait_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0_pingpong_flag));
            uint64_t l1q_offset = 0;
            uint32_t q_load_coeff = 1;
            if (!is_multi_head_mmad) {
                l1q_offset = split_idx * head_split_num_move * round_k + headdim_idx * round_k;
            } else {
                l1q_offset = split_idx * group_num_move * T_BLOCK_SIZE;
                q_load_coeff = cur_head_num_round;
            }
            if (q_load_coeff == 1) {
                pa_l1_to_l0_a_vector<IN_DTYPE, false>(
                    l0a_buf_tensor + (l0_offset),
                    l1q_buf_addr_tensor + (l1q_offset),
                    0,
                    (round_k  + T_CUBE_MATRIX_SIZE - 1) / T_CUBE_MATRIX_SIZE,  // repeat
                    0,
                    1,                                                    // srcStride
                    0,
                    0                                                    // dstStride
                );
            } else {
                for (uint64_t loa_load_idx = 0; loa_load_idx < q_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
                    pa_l1_to_l0_a_vector<IN_DTYPE, false>(
                        l0a_buf_tensor + (l0_offset + loa_load_idx * round_k * BLOCK_SIZE),
                        l1q_buf_addr_tensor + (l1q_offset + loa_load_idx * T_CUBE_MATRIX_SIZE),
                        0,
                        round_k / T_BLOCK_SIZE,            // repeat
                        0,
                        q_load_coeff / BLOCK_SIZE,                            // srcStride
                        0,
                        0                                                     // dstStride
                    );
                }
            }
            uint32_t mad_l0b_offset = 0;
            if (group_num == 1 || tilingKeyType == TilingKeyType::TILING_INT8_CUBE_QUANT) {
                    if (headdim_idx == 0) {
                        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1b_pingpong_flag + 2));
                    }
                if (is_l0b_pingpong_off) {
                    mad_l0b_offset = 0;
                    wait_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
                } else {
                    mad_l0b_offset = l0_offset;
                }
                pa_l1_to_l0_b_vector<IN_DTYPE, false>(
                    l0b_buf_tensor + (mad_l0b_offset),
                    l1kv_buf_addr_tensor + (move_l1b_offset + headdim_idx * round_k * qk_round_n),
                    0,
                    (round_k * qk_round_n) / T_CUBE_MATRIX_SIZE,                   // repeat
                    0,
                    1,                                        // srcStride
                    0,
                    0                                        // dstStride
                );
            } else {
                if (is_l0b_pingpong_off) {
                    l0b_offset = 0;
                    wait_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
                } else if (move_l0b_flag) {
                    l0b_pingpong_flag = 1 - l0b_pingpong_flag;
                    l0b_offset = l0b_pingpong_flag * L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
                    wait_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
                }
                    if (headdim_idx == 0 && move_l1b_flag) {
                        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1b_pingpong_flag + 2));
                    }
                if (move_l0b_flag) {
                    uint64_t l1kv_offset = move_l1b_offset + headdim_idx * round_k * qk_round_n;
                    pa_l1_to_l0_b_vector<IN_DTYPE, false>(
                        l0b_buf_tensor + (l0b_offset),
                        l1kv_buf_addr_tensor + (l1kv_offset),
                        0,
                        round_k * qk_round_n / T_CUBE_MATRIX_SIZE,  // repeat
                        0,
                        1,                                        // srcStride
                        0,
                        0                                        // dstStride
                    );
                }
                mad_l0b_offset = l0b_offset;
            }

            if (headdim_idx == loop_mad - 1) {
                    if ((group_num != 1 && move_l1b_flag) || group_num == 1) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<::event_t>(l1b_pingpong_flag + 2));
                    }
            }

            set_flag(PIPE_MTE1, PIPE_M, static_cast<::event_t>(l0_pingpong_flag));
            wait_flag(PIPE_MTE1, PIPE_M, static_cast<::event_t>(l0_pingpong_flag));
            wait_flag(PIPE_FIX, PIPE_M, static_cast<::event_t>(l0_pingpong_flag));
            mad(
                mm1_l0c_buf_tensor + (l0c_offset),
                l0a_buf_tensor + (l0_offset),
                l0b_buf_tensor + (mad_l0b_offset),
                m,
                __k,
                qk_n,
                0,
                false,
                false,
                1);
            if (is_l0b_pingpong_off) {
                set_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
            } else {
                    if (group_num != 1 && move_l0b_flag) {
                        set_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
                    }
            }
            set_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0_pingpong_flag));
            set_flag(PIPE_M, PIPE_FIX, static_cast<::event_t>(l0_pingpong_flag));
            wait_flag(PIPE_M, PIPE_FIX, static_cast<::event_t>(l0_pingpong_flag));
            // copy S to gm
            uint64_t s_gm_offset = headdim_idx * group_num_move * qk_round_n;
            pa_l0c_to_gm_nd_fp32(
                s_gm_tensor + (s_gm_offset),
                mm1_l0c_buf_tensor + (l0c_offset),
                m,           // MSize
                qk_round_n,  // NSize
                RoundUp<16>(m), // srcStride
                qk_round_n  // dstStride_dst_D
            );
            set_flag(PIPE_FIX, PIPE_M, static_cast<::event_t>(l0_pingpong_flag));
            l0_pingpong_flag = 1 - l0_pingpong_flag;
            l0_offset = l0_pingpong_flag * L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
            l0c_offset = l0_pingpong_flag * L0C_FLOAT_BUF_SIZE;
        }
    }



    __attribute__((always_inline)) inline __aicore__ void ProcessPV(
        __gm__ mm2CopyType *__restrict__ o_tmp_gm_tensor,
        __gm__ IN_DTYPE *__restrict__ p_gm_tensor,
        __cbuf__ IN_DTYPE *__restrict__ l1p_buf_addr_tensor,
        uint32_t qk_n, uint32_t qk_round_n, uint32_t head_num_move, uint32_t group_num_move,
        uint32_t head_split_num_move, uint32_t cur_head_num, uint32_t cur_head_num_round,
        uint32_t split_idx, bool move_l1b_flag, uint32_t softmax_ready_flag, bool is_l0b_pingpong_off)
    {
        uint32_t loop_mad = (group_num == 1) ? head_num_move : 1;
        for (uint32_t headdim_idx = 0; headdim_idx < loop_mad; headdim_idx++) {
            bool move_l0b_flag = move_l1b_flag;
            wait_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0_pingpong_flag));
            uint32_t mad_l0b_offset = 0;
            LoadData2dTransposeParams loadDataParams;
            loadDataParams.dstGap = 0;
            loadDataParams.startIndex = 0;
            loadDataParams.dstFracGap = 0;
            if (group_num == 1 || tilingKeyType == TilingKeyType::TILING_INT8_CUBE_QUANT) {
                    if (headdim_idx == 0) {
                        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1b_pingpong_flag + 2));
                    }
                if (is_l0b_pingpong_off) {
                    mad_l0b_offset = 0;
                    wait_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
                } else {
                    mad_l0b_offset = l0_offset;
                }
                if(qk_round_n <= round_k || tilingKeyType == TilingKeyType::TILING_QUANT_FP16OUT || tilingKeyType == TilingKeyType::TILING_QUANT_BF16OUT) {// Nz -> nZ
                    loadDataParams.repeatTimes = round_k / T_BLOCK_SIZE;
                    loadDataParams.srcStride = qk_round_n / T_BLOCK_SIZE;
                    uint16_t dstGap = sizeof(IN_DTYPE) == 1 ? 1 : 0;
                    loadDataParams.dstGap = dstGap;
                    for (uint32_t l0b_load_idx = 0; l0b_load_idx < qk_round_n / (T_BLOCK_SIZE); ++l0b_load_idx) {
                        load_cbuf_to_cb_transpose(
                            l0b_buf_tensor + (mad_l0b_offset + l0b_load_idx * RoundUp<16>(__k) * T_BLOCK_SIZE),
                            l1kv_buf_addr_tensor + (move_l1b_offset + headdim_idx * round_k * qk_round_n / group_num +
                                l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE),
                            loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                            loadDataParams.dstGap, (addr_cal_mode_t)0, loadDataParams.dstFracGap);
                    }
                } else {
                    loadDataParams.repeatTimes = qk_round_n / T_BLOCK_SIZE;
                    loadDataParams.dstGap = round_k / BLOCK_SIZE - 1;
                    loadDataParams.srcStride = 1;
                    for (uint32_t l0b_load_idx = 0; l0b_load_idx < round_k / T_BLOCK_SIZE; ++l0b_load_idx) {
                        load_cbuf_to_cb_transpose(
                            l0b_buf_tensor + (mad_l0b_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE),
                            l1kv_buf_addr_tensor + (move_l1b_offset + headdim_idx * round_k * qk_round_n / group_num +
                                l0b_load_idx * qk_round_n * T_BLOCK_SIZE),
                            loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                            loadDataParams.dstGap, (addr_cal_mode_t)0, loadDataParams.dstFracGap);
                    }
                }
            } else {
                if (is_l0b_pingpong_off) {
                        l0b_offset = 0;
                        wait_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
                 } else if (move_l0b_flag) {
                        l0b_pingpong_flag = 1 - l0b_pingpong_flag;
                        l0b_offset = l0b_pingpong_flag * L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
                        wait_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
                }
                    if (headdim_idx == 0 && move_l1b_flag) {
                        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1b_pingpong_flag + 2));
                    }
                if (move_l0b_flag) {
                    uint64_t l1kv_offset = move_l1b_offset + headdim_idx * round_k * qk_round_n;
                    if (qk_round_n <= round_k || tilingKeyType == TilingKeyType::TILING_QUANT_FP16OUT || tilingKeyType == TilingKeyType::TILING_QUANT_BF16OUT) {// Nz -> nZ
                        loadDataParams.repeatTimes = round_k / T_BLOCK_SIZE;
                        loadDataParams.srcStride = qk_round_n / T_BLOCK_SIZE;
                        uint16_t dstGap = sizeof(IN_DTYPE) == 1 ? 1 : 0;
                        loadDataParams.dstGap = dstGap;
                        for (uint32_t l0b_load_idx = 0; l0b_load_idx < qk_round_n / T_BLOCK_SIZE; ++l0b_load_idx) {
                            load_cbuf_to_cb_transpose(
                                l0b_buf_tensor + (l0b_offset + l0b_load_idx * RoundUp<16>(__k) * T_BLOCK_SIZE),
                                l1kv_buf_addr_tensor + (l1kv_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE),
                                loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                                loadDataParams.dstGap, (addr_cal_mode_t)0, loadDataParams.dstFracGap);
                        }
                    } else {
                        for (uint32_t l0b_load_idx = 0; l0b_load_idx < round_k / T_BLOCK_SIZE; ++l0b_load_idx) {
                        loadDataParams.repeatTimes = qk_round_n / T_BLOCK_SIZE;
                        loadDataParams.srcStride = 1;
                        loadDataParams.dstGap = round_k / BLOCK_SIZE - 1;
                        load_cbuf_to_cb_transpose(
                            l0b_buf_tensor + (l0b_offset + l0b_load_idx * T_BLOCK_SIZE * T_BLOCK_SIZE),
                            l1kv_buf_addr_tensor + (l1kv_offset + l0b_load_idx * qk_round_n * T_BLOCK_SIZE),
                            loadDataParams.startIndex, loadDataParams.repeatTimes, loadDataParams.srcStride,
                            loadDataParams.dstGap, (addr_cal_mode_t)0, loadDataParams.dstFracGap);
                        }
                    }
                }
                mad_l0b_offset = l0b_offset;
            }

            if (split_idx == 0 && headdim_idx == 0) {
                wait_flag_dev(softmax_ready_flag);
                if (!is_multi_head_mmad) {
                    pa_gm_to_l1_nd_nd<IN_DTYPE>(
                        l1p_buf_addr_tensor,
                        p_gm_tensor,
                        1,
                        0,
                        0,
                        RoundUp<BLOCK_SIZE>(qk_n) * cur_head_num * T_BLOCK_OFFSET,               // lenBurst
                        0,
                        0
                    );
                } else {
                    pa_gm_to_l1_nd_nz<IN_DTYPE>(
                        l1p_buf_addr_tensor,
                        p_gm_tensor,
                        cur_head_num,         // nValue
                        (cur_head_num + 15) / 16 * 16,// dstNzC0Stride
                        0,                     // dstNzMatrixStride, unused
                        qk_round_n,           // dValue
                        0,                     // dstNzMatrixStride, unused
                        RoundUp<BLOCK_SIZE>(qk_n) * T_BLOCK_OFFSET           // srcDValue
                    );
                }
            }

            set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l0_pingpong_flag));
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l0_pingpong_flag));
            uint64_t l1p_offset = 0;
            uint32_t p_load_coeff = 1;
            if (!is_multi_head_mmad) {
                 l1p_offset =  split_idx * head_split_num_move * RoundUp<BLOCK_SIZE>(qk_n)  * T_BLOCK_OFFSET +
                     headdim_idx * RoundUp<BLOCK_SIZE>(qk_n)  * T_BLOCK_OFFSET;
            } else {
                l1p_offset = split_idx * group_num_move * T_BLOCK_SIZE;
                p_load_coeff = cur_head_num_round;
            }
            if (p_load_coeff == 1) {
                pa_l1_to_l0_a_vector<IN_DTYPE, false>(
                    l0a_buf_tensor + (l0_offset),
                    l1p_buf_addr_tensor + (l1p_offset),
                    0,
                    (qk_round_n + T_CUBE_MATRIX_SIZE - 1) / T_CUBE_MATRIX_SIZE,  // repeat
                    0,
                    1,                                                       // srcStride
                    0,
                    0                                                        // dstStride
                );
            } else {
                for (uint64_t loa_load_idx = 0; loa_load_idx < p_load_coeff / BLOCK_SIZE; ++loa_load_idx) {
                    pa_l1_to_l0_a_vector<IN_DTYPE, false>(
                        l0a_buf_tensor + (l0_offset + loa_load_idx * qk_round_n * BLOCK_SIZE),
                        l1p_buf_addr_tensor + (l1p_offset + loa_load_idx * T_CUBE_MATRIX_SIZE),
                        0,
                        qk_round_n / T_BLOCK_SIZE,                                 // repeat
                        0,
                        p_load_coeff / BLOCK_SIZE,                               // srcStride
                        0,
                        0                                                        // dstStride
                    );
                }
            }

            if (headdim_idx == loop_mad - 1) {
                    if (group_num != 1 && move_l1b_flag || group_num == 1) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<::event_t>(l1b_pingpong_flag + 2));
                    }
            }

            set_flag(PIPE_MTE1, PIPE_M, static_cast<::event_t>(l0_pingpong_flag));
            wait_flag(PIPE_MTE1, PIPE_M, static_cast<::event_t>(l0_pingpong_flag));
            wait_flag(PIPE_FIX, PIPE_M, static_cast<::event_t>(l0_pingpong_flag));
            mad(
                mm2_l0c_buf_tensor + (l0c_offset),
                l0a_buf_tensor + (l0_offset),
                l0b_buf_tensor + (mad_l0b_offset),
                m,
                qk_n,
                __k,
                0,
                false,
                false,
                1);

            if (is_l0b_pingpong_off) {
                set_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
            } else {
                    if (group_num != 1 && move_l0b_flag) {
                        set_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0b_pingpong_flag + 2));
                    }
            }
            set_flag(PIPE_M, PIPE_MTE1, static_cast<::event_t>(l0_pingpong_flag));
            set_flag(PIPE_M, PIPE_FIX, static_cast<::event_t>(l0_pingpong_flag));
            wait_flag(PIPE_M, PIPE_FIX, static_cast<::event_t>(l0_pingpong_flag));
            // copy O to gm
            uint64_t o_temp_gm_offset = headdim_idx * group_num_move * RoundUp<16>(__k);
            pa_l0c_to_gm_nd_fp32(
                o_tmp_gm_tensor + (o_temp_gm_offset),
                mm2_l0c_buf_tensor + (l0c_offset),
                m,        // MSize
                RoundUp<16>(__k),  // NSize
                RoundUp<16>(m),       // srcStride
                RoundUp<16>(__k)  // dstStride_dst_D
            );

            set_flag(PIPE_FIX, PIPE_M, static_cast<::event_t>(l0_pingpong_flag));
            ChangeL0PingPongFlag();
        }
    }



    __attribute__((always_inline)) inline __aicore__ void ChangePingPongFlag() {
        l1_pingpong_flag = 1 - l1_pingpong_flag;
        l1_offset = l1_pingpong_flag * L1_UINT8_BUF_SIZE_DECODER / sizeof(IN_DTYPE);
        l1_scale_offset = l1_pingpong_flag * L1_SCALE_UINT64_SIZE;
        l1_bias_offset = l1_pingpong_flag * L1_OFFSET_INT32_SIZE;
        if (group_num == 1) {
            l1b_pingpong_flag = 1 - l1b_pingpong_flag;
            l1b_offset = l1b_pingpong_flag * L1_KV_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
        }
    }

    __attribute__((always_inline)) inline __aicore__ void ChangeL0PingPongFlag() {
        if (is_l0c_pingpong_off) {
            l0_pingpong_flag = 0;
            l0_offset = 0;
            l0c_offset = 0;
        } else {
            l0_pingpong_flag = 1 - l0_pingpong_flag;
            l0_offset = l0_pingpong_flag * L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
            l0c_offset = l0_pingpong_flag * L0C_FLOAT_BUF_SIZE;
        }

    }



    __aicore__ __attribute__((always_inline)) inline void InnerRunCube(uint32_t cur_batch, uint32_t start_head, uint32_t cur_head_num, uint32_t head_split_loop,
                                    uint32_t start_kv, uint32_t cur_kv_seqlen, uint32_t offset_tiling, uint32_t group_num_move, uint32_t head_split_num_move)
    {
        uint32_t addr_q_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 4 + offset_tiling));
        uint32_t addr_q_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 5 + offset_tiling));
        uint64_t addr_q_scalar = (uint64_t)(((uint64_t)addr_q_high32) << 32 | addr_q_loww32);
        uint64_t q_offset = addr_q_scalar + start_head * embedding_size;

        uint32_t pp_n_scalar = block_size_calc;

        uint32_t n_loop = (cur_kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;
        sub_n_loop = pp_n_scalar / block_size;
        real_n_loop = (cur_kv_seqlen + block_size - 1) / block_size;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);

        uint32_t cur_head_num_round = (cur_head_num + 15) / 16 * 16;
        m = (group_num == 1) ? 1 : group_num_move;
        is_multi_head_mmad = (group_num_move > 1) && (tilingKeyType != TilingKeyType::TILING_INT8_CUBE_QUANT);
        bool is_l0b_pingpong_off = (RoundUp<T_BLOCK_SIZE>(block_size_calc) * round_k > (L0AB_UINT8_BUF_SIZE  /  sizeof(IN_DTYPE))) ? 1 : 0;
        for (uint32_t n_idx = 0; n_idx < n_loop; n_idx += 2) {  // for k_seqlen
            if (n_idx == (n_loop - 1)) {
                qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
                qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);
            }
            if ((n_idx + 1) == (n_loop - 1)) {
                qk_n_2 = (cur_kv_seqlen - (n_idx + 1) * pp_n_scalar);
                qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n_2);
            }
            /* ************ CUBE1 stage1  ************* */
            for (uint32_t split_idx = 0; split_idx < head_split_loop; split_idx++) {  // for head
                bool move_l1b_flag = (split_idx == 0) || ((start_head + split_idx * group_num_move) % group_num) == 0;
                // Only need load Q once
                uint32_t head_num_move = ((group_num_move == 1) && (split_idx == (head_split_loop - 1))) ?
                        cur_head_num - head_split_num_move * split_idx * group_num_move : head_split_num_move;
                if (n_idx == 0 && split_idx == 0) {
                    LoadQToL1(q_offset, cur_head_num);
                }

                   set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1_pingpong_flag));
                // *** Prepare K to L1
                uint64_t hiddenSize_offset =
                    (start_head + split_idx * head_split_num_move * group_num_move) / group_num * embedding_size;


                uint64_t deq_scale1_hiddenSize_offset = hiddenSize_offset;
                uint64_t offset1_hiddenSize = k_bias_flag ? hiddenSize_offset : 0;
                    LoadKVToL1(
                        k_gm + (hiddenSize_offset),
                        l1kv_buf_addr_tensor,
                        move_l1b_flag,
                        head_num_move, cur_batch, cur_kv_seqlen, start_kv, qk_round_n,
                        real_n_loop, sub_n_loop, n_idx
                    );
                wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1_pingpong_flag));

                ProcessQK(
                    s_gm + ((uint64_t)block_idx * TMP_SIZE_DECODER +
                         split_idx * head_split_num_move * group_num_move * qk_round_n),
                    qk_n, qk_round_n, head_num_move, group_num_move,
                    head_split_num_move, cur_head_num_round,split_idx, move_l1b_flag, is_l0b_pingpong_off);
                ChangePingPongFlag();
            }
            pipe_barrier(PIPE_FIX);
            DdrBarrierBeforeFfts();
            FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY_DECODER);

            /* ************ CUBE1 stage2  ************* */
            if (n_idx + 1 < n_loop) {
                for (uint32_t split_idx = 0; split_idx < head_split_loop; split_idx++) {  // for head
                    bool move_l1b_flag = (split_idx == 0) || ((start_head + split_idx * group_num_move) % group_num) == 0;
                    // Only need load Q once
                    uint32_t head_num_move = ((group_num_move == 1) && (split_idx == (head_split_loop - 1))) ?
                            cur_head_num - head_split_num_move * split_idx * group_num_move : head_split_num_move;

                        set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1_pingpong_flag));
                    // *** Prepare K to L1
                    uint64_t hiddenSize_offset = (start_head + split_idx * head_split_num_move * group_num_move) / group_num * embedding_size;
                    uint64_t deq_scale1_hiddenSize_offset = hiddenSize_offset;
                    uint64_t offset1_hiddenSize = k_bias_flag ? hiddenSize_offset : 0;
                        LoadKVToL1(
                            k_gm + (hiddenSize_offset),
                            l1kv_buf_addr_tensor,
                            move_l1b_flag,
                            head_num_move, cur_batch, cur_kv_seqlen, start_kv, qk_round_n_2,
                            real_n_loop, sub_n_loop, (n_idx + 1)
                        );
                    wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1_pingpong_flag));
                    // s_gm ping-pong halves are separate (stage-2 uses the second half).
                    ProcessQK(
                        s_gm + ((uint64_t)block_idx * TMP_SIZE_DECODER +
                            split_idx * head_split_num_move * group_num_move * qk_round_n_2 +
                            TMP_SIZE_DECODER / 2),
                        qk_n_2, qk_round_n_2, head_num_move, group_num_move,
                        head_split_num_move, cur_head_num_round, split_idx, move_l1b_flag, is_l0b_pingpong_off);
                    ChangePingPongFlag();
                }
                pipe_barrier(PIPE_FIX);
                DdrBarrierBeforeFfts();
                FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY_STAGE2);
            }

            /* ************ CUBE2 stage1  ************* */
            for (uint32_t split_idx = 0; split_idx < head_split_loop; split_idx++) {
                int32_t head_num_move = ((group_num_move == 1) && (split_idx == (head_split_loop - 1))) ?
                        cur_head_num - head_split_num_move * split_idx * group_num_move : head_split_num_move;
                bool move_l1b_flag = (split_idx == 0) || ((start_head + split_idx * group_num_move) % group_num) == 0;
                // *** Prepare V to L1
                uint64_t hiddenSize_offset =
                    (start_head + split_idx * head_split_num_move * group_num_move) / group_num * embedding_size;

                uint64_t deq_scale2_hiddenSize_offset = hiddenSize_offset;
                uint64_t offset2_hiddenSize = v_bias_flag ? hiddenSize_offset : 0;
                    LoadKVToL1(
                        v_gm + (hiddenSize_offset),
                        l1kv_buf_addr_tensor,
                        move_l1b_flag,
                        head_num_move, cur_batch, cur_kv_seqlen, start_kv, RoundUp<T_BLOCK_SIZE>(qk_round_n),
                        real_n_loop, sub_n_loop, n_idx
                    );
                set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1_pingpong_flag));
                wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1_pingpong_flag));
                ProcessPV(
                    o_tmp_gm + ((uint64_t)block_idx * TMP_SIZE +
                        split_idx * head_split_num_move * group_num_move * RoundUp<16>(__k)),
                    p_gm + ((uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET),
                    l1p_buf_addr_tensor,
                    qk_n, RoundUp<T_BLOCK_SIZE>(qk_round_n), head_num_move, group_num_move,
                    head_split_num_move, cur_head_num, cur_head_num_round,
                    split_idx, move_l1b_flag, SOFTMAX_READY_DECODER, is_l0b_pingpong_off);
                ChangePingPongFlag();
            }
            pipe_barrier(PIPE_FIX);
            DdrBarrierBeforeFfts();
            FftsCrossCoreSync<PIPE_FIX, 2>(UPDATE_READY_DECODER);

            /* ************ CUBE2 stage2  ************* */
            if (n_idx + 1 < n_loop) {
                for (uint32_t split_idx = 0; split_idx < head_split_loop; split_idx++) {
                    int32_t head_num_move = ((group_num_move == 1) && (split_idx == (head_split_loop - 1))) ?
                            cur_head_num - head_split_num_move * split_idx * group_num_move : head_split_num_move;
                    bool move_l1b_flag = (split_idx == 0) || ((start_head + split_idx * group_num_move) % group_num) == 0;
                    // *** Prepare V to L1
                    uint64_t hiddenSize_offset =
                        (start_head + split_idx * head_split_num_move * group_num_move) / group_num * embedding_size;

                    uint64_t deq_scale2_hiddenSize_offset = hiddenSize_offset;
                    uint64_t offset2_hiddenSize = v_bias_flag ? hiddenSize_offset : 0;
                        LoadKVToL1(
                            v_gm + (hiddenSize_offset),
                            l1kv_buf_addr_tensor,
                            move_l1b_flag,
                            head_num_move, cur_batch, cur_kv_seqlen, start_kv,  RoundUp<T_BLOCK_SIZE>(qk_round_n_2),
                            real_n_loop, sub_n_loop, (n_idx + 1)
                        );
                    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1_pingpong_flag));
                    wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(l1_pingpong_flag));
                    ProcessPV(
                        o_tmp_gm + ((uint64_t)block_idx * TMP_SIZE +
                            split_idx * head_split_num_move * group_num_move * RoundUp<16>(__k) +
                            TMP_SIZE / 2),
                        p_gm + ((uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET  + TMP_SIZE * T_BLOCK_OFFSET / 2),
                        l1p_buf_addr_tensor + (qk_round_n * cur_head_num_round),
                        qk_n_2, RoundUp<T_BLOCK_SIZE>(qk_round_n_2), head_num_move, group_num_move,
                        head_split_num_move, cur_head_num, cur_head_num_round,
                        split_idx, move_l1b_flag, SOFTMAX_READY_STAGE2, is_l0b_pingpong_off);
                    ChangePingPongFlag();
                }
                pipe_barrier(PIPE_FIX);
                DdrBarrierBeforeFfts();
                FftsCrossCoreSync<PIPE_FIX, 2>(UPDATE_READY_STAGE2);
            }
        }
    }


private:
    __gm__ IN_DTYPE *__restrict__ q_gm{nullptr};
    __gm__ IN_KVDTYPE *__restrict__ k_gm{nullptr};
    __gm__ IN_KVDTYPE *__restrict__ v_gm{nullptr};


    __gm__ mm1CopyType *__restrict__ s_gm{nullptr};
    __gm__ IN_DTYPE *__restrict__ p_gm{nullptr};
    __gm__ mm2CopyType *__restrict__ o_tmp_gm{nullptr};
    __gm__ int32_t *__restrict__ block_tables_gm{nullptr};
    __gm__ uint8_t *__restrict__ tiling_gm{nullptr};

    __gm__ IN_DTYPE *gm_k16_ping_{nullptr};
    __gm__ IN_DTYPE *gm_k16_pong_{nullptr};
    __gm__ IN_DTYPE *gm_v16_ping_{nullptr};
    __gm__ IN_DTYPE *gm_v16_pong_{nullptr};

    const uint32_t l1q_buf_addr_offset = 0;

    const uint32_t l1p_buf_addr_offset = (5 * L0AB_UINT8_BLOCK_SIZE);
    const uint32_t l1kv_buf_addr_offset = (7 * L0AB_UINT8_BLOCK_SIZE);

    __cbuf__ IN_DTYPE *l1q_buf_addr_tensor =
        reinterpret_cast<__cbuf__ IN_DTYPE *>((uintptr_t)l1q_buf_addr_offset);
    __cbuf__ IN_DTYPE *l1kv_buf_addr_tensor =
        reinterpret_cast<__cbuf__ IN_DTYPE *>((uintptr_t)l1kv_buf_addr_offset);
    __cbuf__ IN_DTYPE *l1p_buf_addr_tensor =
        reinterpret_cast<__cbuf__ IN_DTYPE *>((uintptr_t)l1p_buf_addr_offset);
    __ca__ IN_DTYPE *l0a_buf_tensor = reinterpret_cast<__ca__ IN_DTYPE *>((uintptr_t)0);
    __cb__ IN_DTYPE *l0b_buf_tensor = reinterpret_cast<__cb__ IN_DTYPE *>((uintptr_t)0);
    __cc__ mm1OutputType *mm1_l0c_buf_tensor = reinterpret_cast<__cc__ mm1OutputType *>((uintptr_t)0);
    __cc__ mm2OutputType *mm2_l0c_buf_tensor = reinterpret_cast<__cc__ mm2OutputType *>((uintptr_t)0);
    __cc__ int32_t *l0c_buf_int32_tensor = reinterpret_cast<__cc__ int32_t *>((uintptr_t)0);


    uint32_t k_bias_flag{0};
    uint32_t v_bias_flag{0};
    uint32_t num_tokens{0};
    uint32_t q_heads{0};
    uint32_t kv_heads{0};
    uint32_t embedding_size{0};
    uint32_t embedding_size_v{0};
    uint32_t block_size{0};
    uint32_t max_num_blocks_per_query{0};
    uint32_t group_num{0};
    uint32_t former_group_num_move{1};
    uint32_t tail_group_num_move{1};
    uint32_t former_head_split_num{1};
    uint32_t tail_head_split_num{1};
    uint32_t stride_kv{0};
    uint32_t stride_vo{0};
    uint32_t m{0};
    uint32_t __k{0};
    uint32_t __v{0};
    uint32_t round_k{0};
    uint32_t round_v{0};
    uint32_t core_per_batch{0};
    uint32_t process_num{0};
    uint64_t former_batch{0};
    uint32_t former_head_split{0};
    uint64_t tail_batch{0};
    uint32_t tail_head_split{0};
    uint32_t head_split_num{0};
    uint32_t tiling_head_size{0};
    uint32_t tiling_para_size{0};
    uint32_t kv_split_per_core{0};
    uint32_t kv_split_core_num{1};
    uint32_t block_size_calc{0};

    uint32_t embed_split_size_qk{0};
    uint32_t embed_split_loop_qk{1};
    uint32_t embed_split_size_v{0};
    uint32_t embed_split_loop_v{1};
    bool is_multi_head_mmad{0};
    uint32_t move_l1b_offset = 0;
    uint32_t q_head_original{0};
    uint32_t compressHead{0};
    uint32_t block_idx{0};
    uint32_t block_num{1};

    uint32_t l1_pingpong_flag = 0;
    uint32_t l1b_pingpong_flag = 0;
    uint32_t l0_pingpong_flag = 0;
    uint32_t l0b_pingpong_flag = 0;
    uint32_t l1p_pingpong_flag = 0;
    uint32_t block_size_inner_count{0};
    uint32_t sub_n_loop{0};
    uint32_t real_n_loop{0};

    uint32_t l1_offset = l1_pingpong_flag * L1_UINT8_BUF_SIZE_DECODER / sizeof(IN_DTYPE);
    uint32_t l1b_offset = l1b_pingpong_flag * L1_KV_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
    uint32_t l1_scale_offset = l1_pingpong_flag * L1_SCALE_UINT64_SIZE;
    uint32_t l1_bias_offset = l1_pingpong_flag * L1_OFFSET_INT32_SIZE;
    uint32_t l0_offset = l0_pingpong_flag * L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
    uint32_t l0c_offset = l0_pingpong_flag * L0C_FLOAT_BUF_SIZE;
    uint32_t l0b_offset = l0b_pingpong_flag * L0AB_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
    uint32_t l1p_start_offset = l1p_pingpong_flag * L1_P_UINT8_BUF_SIZE / sizeof(IN_DTYPE);
    bool is_l0c_pingpong_off = 0;
    uint32_t prefill_batch_size_;
    uint32_t decoder_batch_size_;

};
#elif __DAV_C220_VEC__
enum class ScaleType {
        SCALE_TOR = 0,
        SCALE_LOGN = 1,
        SCALE_LOGN_FP32 = 2
};

template <TilingKeyType tilingKeyType = TilingKeyType::TILING_HALF_DATA, typename IN_DTYPE = half, typename OUT_DTYPE = half, bool SplitKV = false, PagedAttnVariant pagedAttnVariant = PagedAttnVariant::DEFAULT, CompressType compressType = CompressType::COMPRESS_TYPE_UNDEFINED>
class UnpadAttentionDecoderAiv{
public:
    using mm1OutputType = typename AttentionType<tilingKeyType>::mm1OutputType;
    using mm1CopyType = typename AttentionType<tilingKeyType>::mm1CopyType;
    using mmBiasType = typename AttentionType<tilingKeyType>::mmBiasType;
    using mmScaleType = typename AttentionType<tilingKeyType>::mmScaleType;
    using mm2OutputType = typename AttentionType<tilingKeyType>::mm2OutputType;
    using mm2CopyType = typename AttentionType<tilingKeyType>::mm2CopyType;
    static constexpr uint32_t T_BLOCK_SIZE =  BLOCK_SIZE_32 / sizeof(IN_DTYPE);
    static constexpr uint32_t T_BLOCK_OFFSET = 2 / sizeof(IN_DTYPE);

    __aicore__ __attribute__((always_inline)) inline UnpadAttentionDecoderAiv(uint32_t prefill_batch_size, uint32_t decoder_batch_size) {
        prefill_batch_size_ = prefill_batch_size;
        decoder_batch_size_ = decoder_batch_size;
    }

    __aicore__ __attribute__((always_inline)) inline void SetArgs(
        __gm__ uint8_t *__restrict__ sync,
        __gm__ uint8_t* __restrict__ gm_k8,
        __gm__ uint8_t* __restrict__ gm_v8,
        __gm__ uint8_t* __restrict__ gm_scale1,
        __gm__ uint8_t* __restrict__ gm_offset1,
        __gm__ uint8_t* __restrict__ gm_scale2,
        __gm__ uint8_t* __restrict__ gm_offset2,
        __gm__ uint8_t* __restrict__ gm_block_table,
        __gm__ uint8_t *__restrict__ mask_in_gm,
        __gm__ uint8_t *__restrict__ o_out_gm,
        __gm__ uint8_t *__restrict__ s_out_gm,
        __gm__ uint8_t *__restrict__ p_out_gm,
        __gm__ uint8_t *__restrict__ o_temp_gm,
        __gm__ uint8_t *__restrict__ globalo_gm,
        __gm__ uint8_t *__restrict__ o_core_out_tmp_gm,
        __gm__ uint8_t *__restrict__ l_in_gm,
        __gm__ uint8_t* __restrict__ gm_k16,
        __gm__ uint8_t* __restrict__ gm_v16,
        __gm__ uint8_t *__restrict__ tiling_para_gm,
        __gm__ uint8_t *__restrict__ razorOffset,
        __gm__ uint8_t *__restrict__ logN_in_gm,
        uint32_t pto_block_idx,
        uint32_t pto_block_num,
        uint32_t pto_sub_block_id)
    {
        if (sync != nullptr) {
            set_ffts_base_addr((uint64_t)sync);
        }
        block_idx = pto_block_idx;
        block_num = pto_block_num;
        sub_block_idx = static_cast<uint64_t>(pto_sub_block_id);
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        mask_gm = reinterpret_cast<__gm__ OUT_DTYPE *>(mask_in_gm);
        o_gm = reinterpret_cast<__gm__ OUT_DTYPE *>(o_out_gm);
        s_gm = reinterpret_cast<__gm__ mm1CopyType *>(s_out_gm);
        p_gm = reinterpret_cast<__gm__ IN_DTYPE *>(p_out_gm);
        o_tmp_gm = reinterpret_cast<__gm__ mm2CopyType *>(o_temp_gm);
        go_gm = reinterpret_cast<__gm__ float *>(globalo_gm);
        o_core_tmp_gm = reinterpret_cast<__gm__ float *>(o_core_out_tmp_gm);
        tiling_gm = reinterpret_cast<__gm__ uint8_t *>(tiling_para_gm);
        l_gm = reinterpret_cast<__gm__ float *>(l_in_gm);
        gm_block_tables_ = reinterpret_cast<__gm__ int32_t*>(gm_block_table);
        logN_gm = reinterpret_cast<__gm__ float *>(logN_in_gm);
        num_tokens = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm));
        q_heads = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_NUMHEADS));
        embedding_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADDIM));
        embedding_size_v = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADDIM_V));
        block_size = (int32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE));
        max_num_blocks_per_query = (uint32_t)(*((__gm__ uint32_t*)tiling_para_gm + TILING_MAXBLOCKS));
        tor = (float)(*((__gm__ float *)tiling_para_gm + TILING_TOR));
        num_kv_heads = (uint32_t)(*((__gm__ uint32_t*)tiling_para_gm + TILING_KVHEADS));
        former_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_FORMER_BATCH));
        former_head_split = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_FORMER_HEAD));
        tail_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_TAIL_BATCH));
        tail_head_split = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_TAIL_HEAD));
        max_context_len = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MASK_MAX_LEN));
        batch_stride = (uint64_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BATCH_STRIDE));
        head_stride = (uint64_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEAD_STRIDE));
        tiling_head_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_HEADSIZE));
        tiling_para_size = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_PARASIZE));
        group_num = (uint32_t)(*((__gm__ uint32_t*)tiling_para_gm + TILING_GROUPNUM));

        kv_split_per_core = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_KVSPLIT));
        kv_split_core_num = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_KVCORENUM));
        block_size_calc = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_BLOCKSIZE_CALC));
        q_head_original = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_QHEADORIGINAL));
        compressHead = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_COMPRESSHEAD));
        scaleType = (ScaleType)(*((__gm__ uint32_t *)tiling_para_gm + TILING_SCALETYPE));

            former_group_num_move = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_FORMER_GROUP_MOVE));
            tail_group_num_move = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_TAIL_GROUP_MOVE));

        go_flag_scalar = 1;
        gl_flag_scalar = 1;

        modCoef = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_MODCOEF));
        divCoef = (uint32_t)(*((__gm__ uint32_t *)tiling_para_gm + TILING_DIVCOEF));

        __k = embedding_size;
        round_k = RoundUp<T_BLOCK_SIZE>(__k);

        core_per_batch = (q_heads + split_size - 1) / split_size;
        process_num = num_tokens * core_per_batch;
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        core_per_batch = (q_heads + former_head_split - 1) / former_head_split;
        process_num = static_cast<uint64_t>(former_batch) * core_per_batch * kv_split_core_num;
        for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {
            uint32_t cur_batch = process / (core_per_batch * kv_split_core_num) + prefill_batch_size_;
            uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
            cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 13 + offset_tiling));
            offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
            uint32_t batch_idx = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 8 + offset_tiling));
            uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
            tor = (float)(*((__gm__ float *)tiling_gm + TILING_TOR));
            if(scaleType == ScaleType::SCALE_LOGN_FP32) {
                float tor_logN = (float)(*((__gm__ float *)logN_gm + cur_batch));
                tor = tor * tor_logN;
            }
            uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t cur_head = (process / kv_split_core_num) % core_per_batch;
            uint32_t cur_nIndx = process % kv_split_core_num;
            uint32_t start_head = cur_head * former_head_split;
            uint32_t cur_kv_seqlen = kv_split_per_core;
            uint32_t kv_loop = (kv_seqlen_align + kv_split_per_core - 1) /  kv_split_per_core;
            if (cur_nIndx >= kv_loop) {
                continue;
            }
            if (cur_nIndx == (kv_loop - 1)) {
                cur_kv_seqlen = kv_seqlen - cur_nIndx * kv_split_per_core;
            }
            uint32_t cur_head_num = former_head_split;
            if (cur_head == (core_per_batch - 1)) {
                cur_head_num = q_heads - cur_head * former_head_split;
            }
            InnerRunVector(batch_idx, start_head, cur_nIndx, cur_kv_seqlen, cur_head_num, offset_tiling, kv_seqlen, embed_split_size_v_former, embed_split_loop_v_former);
        }
        if (tail_batch > 0) {
            core_per_batch = (q_heads + tail_head_split - 1) / tail_head_split;
            process_num = static_cast<uint64_t>(tail_batch) * core_per_batch;
            for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {
                uint32_t cur_batch = process / core_per_batch + former_batch + prefill_batch_size_;
                uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
                cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 13 + offset_tiling));
                offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
                uint32_t batch_idx = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 8 + offset_tiling));
                uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
                if (kv_seqlen == 0) {
                    continue;
                }
                tor = (float)(*((__gm__ float *)tiling_gm + TILING_TOR));
                if(scaleType == ScaleType::SCALE_LOGN_FP32) {
                    float tor_logN = (float)(*((__gm__ float *)logN_gm + cur_batch));
                    tor = tor * tor_logN;
                }
                uint32_t cur_kv_seqlen = kv_seqlen;
                uint32_t cur_nIndx = 0;
                uint32_t cur_head = process % core_per_batch;
                uint32_t cur_head_num = tail_head_split;
                if (cur_head == (core_per_batch - 1)) {
                    cur_head_num = q_heads - cur_head * tail_head_split;
                }
                uint32_t start_head = (process % core_per_batch) * tail_head_split;
                InnerRunVector(batch_idx, start_head, cur_nIndx, cur_kv_seqlen, cur_head_num, offset_tiling, kv_seqlen, embed_split_size_v_tail, embed_split_loop_v_tail);
            }
        }
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        pipe_barrier(PIPE_ALL);
        if (SplitKV) {
            DdrBarrierBeforeFfts();
            FftsCrossCoreSync<PIPE_MTE3, (uint8_t)0>(REDUCE_READY_DECODER);
            wait_flag_dev(REDUCE_READY_DECODER);
            CombineScale(decoder_batch_size_, q_heads, kv_split_core_num, embedding_size);
        }
    }


private:

    __aicore__ __attribute__((always_inline)) inline void CopyScale(uint32_t sub_m, uint32_t l_offset, uint32_t o_offset)
    {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        copy_ubuf_to_gm_align_b32(
            l_gm + ((int64_t)l_offset),
            tv32_ubuf_tensor,
            0,               // sid
            sub_m,           // nBurst
            4,               // lenBurst
            0,               // leftPaddingNum
            0,               // rightPaddingNum
            0,                 // srcGap
            (kv_split_core_num - 1) * 4 // dstGap
        );
        if (gl_flag_scalar == 0) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            gl_flag_scalar = 1;
        }
        uint32_t src_gap = ((__k % 16 <= 8) && (__k % 16 > 0))? 1 : 0;
        copy_ubuf_to_gm_align_b32(
            o_core_tmp_gm + ((int64_t)o_offset),
            go32_ubuf_tensor,
            0,        // sid
            sub_m,    // nBurst
            __k * 4,  // lenBurst
            0,        // leftPaddingNum
            0,        // rightPaddingNum
            src_gap,   // srcGap
            (kv_split_core_num - 1) * __k * 4  // dstGap
        );
    }
    __aicore__ __attribute__((always_inline)) inline void CombineScale(uint32_t num_tokens, uint32_t q_heads, uint32_t kv_split_core_num, uint32_t embedding_size)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        const uint32_t ll_ubuf_stage2_offset = 0;  // Tile: per-split log-sum L in fp32
        const uint32_t lm_ubuf_stage2_offset = 1 * STAGE2_UB_UINT8_BLOCK_SIZE;  // Tile: row-wise l max in fp32
        const uint32_t tl_ubuf_stage2_offset = 1 * STAGE2_UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE; // Tile: tmp shifted L before exp, fp32
        const uint32_t rs_ubuf_stage2_offset = 2 * STAGE2_UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE; // Tile: row sum after exp, fp32
        const uint32_t ts_ubuf_stage2_offset = 2 * STAGE2_UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE; // Tile: log(row sum) + l_max scratch, fp32
        const uint32_t gl_ubuf_stage2_offset = 2 * STAGE2_UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE; // Tile: global combine scale in fp32
        const uint32_t lo_ubuf_stage2_offset = 4 * STAGE2_UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE;
        const uint32_t to_ubuf_stage2_offset = 8 * STAGE2_UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE;
        const uint32_t go_ubuf_stage2_offset = 12 * STAGE2_UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE;
        const uint32_t go16_ubuf_stage2_offset = 16 * STAGE2_UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE;

        __ubuf__ float *ll_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)ll_ubuf_stage2_offset);
        __ubuf__ float *lm_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)lm_ubuf_stage2_offset);
        __ubuf__ float *tl_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)tl_ubuf_stage2_offset);
        __ubuf__ float *rs_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)rs_ubuf_stage2_offset);
        __ubuf__ float *ts_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)ts_ubuf_stage2_offset);
        __ubuf__ float *gl_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)gl_ubuf_stage2_offset);
        __ubuf__ float *lo_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)lo_ubuf_stage2_offset);
        __ubuf__ float *to_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)to_ubuf_stage2_offset);
        __ubuf__ float *go_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ float *>((uintptr_t)go_ubuf_stage2_offset);
        __ubuf__ OUT_DTYPE *go16_ubuf_stage2_tensor =
            reinterpret_cast<__ubuf__ OUT_DTYPE *>((uintptr_t)go16_ubuf_stage2_offset);

        uint32_t batch_size = num_tokens;
        uint32_t split_block = 1;
        uint32_t __k0 = embedding_size;
        uint32_t roundk_64 = (__k0 + 63) / 64 * 64;
        uint32_t roundk_8 = (__k0 + 7) / 8 * 8;
        uint32_t core_per_batch = (q_heads + split_block - 1) / split_block;

        uint32_t process_num = core_per_batch * batch_size;
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)){
            uint32_t cur_batch = process / core_per_batch + prefill_batch_size_;
            uint32_t offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
            cur_batch = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 13 + offset_tiling));
            offset_tiling = tiling_head_size + tiling_para_size * cur_batch;
            uint32_t kv_seqlen = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 1 + offset_tiling));
            if (kv_seqlen == 0) {
                continue;
            }
            uint32_t addr_o_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 6 + offset_tiling));
            uint32_t addr_o_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 7 + offset_tiling));
            uint64_t addr_o_scalar = (uint64_t)(((uint64_t)addr_o_high32) << 32 | addr_o_loww32);
            uint32_t addr_o_fd_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 15 + offset_tiling));
            uint32_t addr_o_fd_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 16 + offset_tiling));
            uint64_t addr_o_fd_scalar = (uint64_t)(((uint64_t)addr_o_fd_high32) << 32 | addr_o_fd_loww32);
            uint32_t addr_l_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 11 + offset_tiling));
            uint32_t addr_l_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 12 + offset_tiling));
            uint64_t addr_l_scalar = (uint64_t)(((uint64_t)addr_l_high32) << 32 | addr_l_loww32);

            uint32_t kv_seqlen_align = (kv_seqlen + block_size - 1) / block_size * block_size;
            uint32_t m_split = (kv_seqlen_align + kv_split_per_core - 1) /  kv_split_per_core;
            uint32_t cur_core = process % core_per_batch;
            uint32_t cur_head_num = split_block; // Number of query heads processed in this split
            if (cur_core == (core_per_batch - 1)){
                cur_head_num = q_heads - cur_core * split_block;
            }
            uint32_t start_head = cur_core * split_block;
            uint64_t addr_l_offset = addr_l_scalar;
            uint64_t addr_o_offset = addr_o_fd_scalar * kv_split_core_num;
            uint32_t l_remain = m_split % FLOAT_BLOCK_SIZE;
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_ubuf_align_b32(
                ll_ubuf_stage2_tensor,
                l_gm + (addr_l_offset + start_head * kv_split_core_num),
                0,                            // sid
                1,                            // nBurst
                m_split * 4,                  // lenBurst
                0,                           // leftPaddingNum
                FLOAT_BLOCK_SIZE - l_remain,  // rightPaddingNum
                0,                           // srcGap
                0   // dstGap
            );

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            __set_mask(m_split);
            vcmax((lm_ubuf_stage2_tensor), (ll_ubuf_stage2_tensor),
                (m_split + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                static_cast<Order_t>(ORDER_ONLY_VALUE)
            );
            pipe_barrier(PIPE_V);

            // lse_accum - lse_max
            set_flag(PIPE_V, PIPE_S, EVENT_ID3);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID3);
            float lse_max = -(float)(*lm_ubuf_stage2_tensor);
            set_flag(PIPE_S, PIPE_V, EVENT_ID2);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
            vadds((tl_ubuf_stage2_tensor), (ll_ubuf_stage2_tensor), lse_max,
                (m_split + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                8
            );
            pipe_barrier(PIPE_V);

            // expf
            vexp((tl_ubuf_stage2_tensor), (tl_ubuf_stage2_tensor),
                (m_split + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                8
            );
            pipe_barrier(PIPE_V);

            // rowsum lse_sum
            vcadd((rs_ubuf_stage2_tensor), (tl_ubuf_stage2_tensor),
                (m_split + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                0
            );
            pipe_barrier(PIPE_V);
            __set_mask(cur_head_num);
            vln((rs_ubuf_stage2_tensor), (rs_ubuf_stage2_tensor),
                (cur_head_num + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                8
            );
            pipe_barrier(PIPE_V);

            // logf(lse_sum) + lse_max
            vadd((ts_ubuf_stage2_tensor), (rs_ubuf_stage2_tensor), (lm_ubuf_stage2_tensor),
                (cur_head_num + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                1,
                8,
                8,
                8
            );
            pipe_barrier(PIPE_V);

            // scale = expf(lse_accum(l) - lse_logsum)
            __set_mask(m_split);
            set_flag(PIPE_V, PIPE_S, EVENT_ID3);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID3);
            float log_sum = -(float)(*ts_ubuf_stage2_tensor);
            set_flag(PIPE_S, PIPE_V, EVENT_ID2);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
            vadds((gl_ubuf_stage2_tensor), (ll_ubuf_stage2_tensor), log_sum,
                (m_split + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                8
            );
            pipe_barrier(PIPE_V);

            __set_mask(m_split);
            vexp((gl_ubuf_stage2_tensor), (gl_ubuf_stage2_tensor),
                (m_split + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                8
            );
            pipe_barrier(PIPE_V);
            // msplit * 1 * embedding
            copy_gm_to_ubuf_align_b32(
                lo_ubuf_stage2_tensor,
                o_core_tmp_gm + (addr_o_offset + start_head * kv_split_core_num * __k0),
                0,                                           // sid
                m_split,                                     // nBurst
                __k0 * 4,                                    // lenBurst
                0,                                           // leftPaddingNum
                0,                                           // rightPaddingNum
                0,                                           // srcGap
                0                                            // dstGap
            );
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            for (uint32_t n_idx = 0; n_idx < m_split; n_idx++){
                set_flag(PIPE_V, PIPE_S, EVENT_ID3);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID3);
                float scale = (float)(*(gl_ubuf_stage2_tensor + n_idx));
                set_flag(PIPE_S, PIPE_V, EVENT_ID2);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID2);

                vmuls((to_ubuf_stage2_tensor), (lo_ubuf_stage2_tensor + (n_idx * roundk_8)), scale,
                (roundk_64 + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                8
            );
                pipe_barrier(PIPE_V);

                if (n_idx == 0){
                    vadds((go_ubuf_stage2_tensor), (to_ubuf_stage2_tensor), 0,
                roundk_64 / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                8
            );
                    pipe_barrier(PIPE_V);

                }
                else{
                    vadd((go_ubuf_stage2_tensor), (to_ubuf_stage2_tensor), (go_ubuf_stage2_tensor),
                roundk_64 / FLOAT_VECTOR_SIZE,
                1,
                1,
                1,
                8,
                8,
                8
            );
                    pipe_barrier(PIPE_V);

                }
            }
            conv_v<float, OUT_DTYPE>(go16_ubuf_stage2_tensor,
                go_ubuf_stage2_tensor,
                roundk_64 / FLOAT_VECTOR_SIZE,   // repeat
                1,                               // dstBlockStride
                1,                               // srcBlockStride
                4,                               // dstRepeatStride
                8                                // srcRepeatStride
            );
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            copy_ubuf_to_gm_align_b16(
                o_gm + (addr_o_scalar + start_head * __k0),
                go16_ubuf_stage2_tensor,
                0,                       // sid
                1,                       // nBurst
                __k0 * 2,                // lenBurst
                0,                       // leftPaddingNum
                0,                       // rightPaddingNum
                0,                       // srcGap
                0                        // dstGap
            );
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }

    __aicore__ __attribute__((always_inline)) inline void AddMask(
        __gm__ OUT_DTYPE *__restrict__ mask_gm_tensor,
        __ubuf__ OUT_DTYPE *__restrict__ mask_ubuf_tensor,
        __ubuf__ float *__restrict__ mask32_ubuf_tensor,
        uint32_t sub_m,
        uint32_t qk_n,
        uint32_t qk_round_n,
        uint32_t mask_offset)
    {
        uint32_t mask_repeat_stride = head_stride == 0 ? 0 : qk_round_n / FLOAT_BLOCK_SIZE;
        uint32_t mask_nburst = head_stride == 0 ? 1 : sub_m;
        copy_gm_to_ubuf_align_b16(
            mask_ubuf_tensor,
            mask_gm_tensor,
            0,                                 // sid
            mask_nburst,                             // nBurst
            qk_n * 2,                          // lenBurst
            0,                                 // leftPaddingNum
            0,                                 // rightPaddingNum
            (max_context_len - qk_n) * 2,      // srcGap
            0                                  // dstGap
        );
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        conv_v<OUT_DTYPE, float>(mask32_ubuf_tensor,
            mask_ubuf_tensor,
            (mask_nburst * qk_round_n + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
            1,                                                         // dstBlockStride
            1,                                                         // srcBlockStride
            8,                                                         // dstRepeatStride
            4                                                          // srcRepeatStride
        );
        pipe_barrier(PIPE_V);
        // *** ls = ls + mask
        if (qk_round_n  > FLOAT_BLOCK_SIZE * 255) {
            for (uint32_t vadd_idx = 0; vadd_idx < sub_m; ++vadd_idx){
                vadd((ls32_ubuf_tensor + (vadd_idx * qk_round_n)), (ls32_ubuf_tensor + (vadd_idx * qk_round_n)), (mask32_ubuf_tensor + (vadd_idx * mask_repeat_stride * FLOAT_BLOCK_SIZE)),
                qk_n / FLOAT_VECTOR_SIZE,
                1,
                1,
                1,
                8,
                8,
                8
            );
            }
            if (qk_n % FLOAT_VECTOR_SIZE > 0) {
                uint32_t offset = qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE;
                __set_mask(qk_n % FLOAT_VECTOR_SIZE);
                for (uint32_t vadd_idx = 0; vadd_idx < sub_m; ++vadd_idx) {
                    vadd((ls32_ubuf_tensor + (vadd_idx * qk_round_n + offset)), (ls32_ubuf_tensor + (vadd_idx * qk_round_n + offset)), (mask32_ubuf_tensor + (vadd_idx * qk_round_n + offset)),
                1,
                1,
                1,
                1,
                1,
                1,
                1
            );
                }
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }
        } else {
            for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
                vadd((ls32_ubuf_tensor + (vadd_idx * FLOAT_VECTOR_SIZE)), (ls32_ubuf_tensor + (vadd_idx * FLOAT_VECTOR_SIZE)), (mask32_ubuf_tensor + (vadd_idx * FLOAT_VECTOR_SIZE)),
                sub_m,
                1,
                1,
                1,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                mask_repeat_stride
            );
            }
            if (qk_n % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(qk_n % FLOAT_VECTOR_SIZE);
                vadd((ls32_ubuf_tensor + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (ls32_ubuf_tensor + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (mask32_ubuf_tensor + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)),
                sub_m,
                1,
                1,
                1,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                mask_repeat_stride
            );
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }
        }
        pipe_barrier(PIPE_V);
    }


   __aicore__ __attribute__((always_inline)) inline void ReduceMaxRepeatM(
        __ubuf__ float *__restrict__ dst,
        __ubuf__ float *__restrict__ src,
        __ubuf__ float *__restrict__ tempTensor,
        uint32_t sub_m,
        uint32_t qk_n,
        uint32_t qk_round_n)
    {
        if (qk_n <= FLOAT_VECTOR_SIZE) {
            __set_mask(qk_n);
            vcmax((dst), (src),
                sub_m,
                1,
                1,
                qk_round_n / FLOAT_BLOCK_SIZE,
                static_cast<Order_t>(ORDER_ONLY_VALUE)
            );
        } else {
            copy_ubuf_to_ubuf(
                tempTensor,
                src,
                0,                                             // sid
                sub_m,                                         // nBurst
                HALF_VECTOR_SIZE / BLOCK_SIZE,                 // lenBurst
                (qk_round_n - FLOAT_VECTOR_SIZE) / FLOAT_BLOCK_SIZE,  // srcGap
                0                                              // dstGap
            );
            pipe_barrier(PIPE_V);
            for (uint32_t rowmax_idx = 1; rowmax_idx < qk_n / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                vmax((tempTensor), (tempTensor), (src + (rowmax_idx * FLOAT_VECTOR_SIZE)),
                sub_m,
                1,
                1,
                1,
                8,
                8,
                qk_round_n / FLOAT_BLOCK_SIZE
            );
            pipe_barrier(PIPE_V);
            }
            if (qk_n % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(qk_n % FLOAT_VECTOR_SIZE);
                vmax((tempTensor), (tempTensor), (src + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)),
                sub_m,
                1,
                1,
                1,
                8,
                8,
                qk_round_n / FLOAT_BLOCK_SIZE
            );
            }
            pipe_barrier(PIPE_V);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            vcmax((dst), (tempTensor),
                sub_m,
                1,
                1,
                8,
                static_cast<Order_t>(ORDER_ONLY_VALUE)
            );
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);
    }

    __aicore__ __attribute__((always_inline)) inline void ReduceSumRepeatM(
        __ubuf__ float *__restrict__ dst,
        __ubuf__ float *__restrict__ src,
        uint32_t sub_m,
        uint32_t qk_n,
        uint32_t qk_round_n)
    {
        if (qk_n <= FLOAT_VECTOR_SIZE) {
            __set_mask(qk_n);
            vcadd((dst), (src),
                sub_m,
                1,
                1,
                qk_round_n / FLOAT_BLOCK_SIZE,
                0
            );
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        } else {
            for (uint32_t rowsum_idx = 1; rowsum_idx < qk_n / FLOAT_VECTOR_SIZE; ++rowsum_idx) {
                vadd((src), (src), (src + (rowsum_idx * FLOAT_VECTOR_SIZE)),
                sub_m,
                1,
                1,
                1,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE
            );
                pipe_barrier(PIPE_V);
            }
            if (qk_n % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(qk_n % FLOAT_VECTOR_SIZE);
                vadd((src), (src), (src + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)),
                sub_m,
                1,
                1,
                1,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE
            );
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }
            pipe_barrier(PIPE_V);

            vcadd((dst), (src),
                sub_m,
                1,
                1,
                qk_round_n / FLOAT_BLOCK_SIZE,
                0
            );
        }
    }

    __aicore__ __attribute__((always_inline)) inline void TensorSubValueRepeatM(
        __ubuf__ float *__restrict__ dst,
        __ubuf__ float *__restrict__ src,
        __ubuf__ float *__restrict__ MaxTensor,
        __ubuf__ float *__restrict__ tempMaxTensor,
        uint32_t sub_m,
        uint32_t round_sub_m,
        uint32_t qk_n,
        uint32_t qk_round_n)
    {
        vbrcb((__ubuf__ uint32_t *)tempMaxTensor, (__ubuf__ uint32_t *)MaxTensor,
            1,
            8,
            round_sub_m / FLOAT_BLOCK_SIZE
        );
        pipe_barrier(PIPE_V);
        for (uint32_t sub_v_idx = 0; sub_v_idx < qk_n / FLOAT_VECTOR_SIZE; ++sub_v_idx) {
            vsub((dst + (sub_v_idx * FLOAT_VECTOR_SIZE)), (src + (sub_v_idx * FLOAT_VECTOR_SIZE)), (tempMaxTensor),
                sub_m,
                1,
                1,
                0,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                1
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            vsub((dst + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (src + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (tempMaxTensor),
                sub_m,
                1,
                1,
                0,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                1
            );
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ __attribute__((always_inline)) inline void TensorDivRepeatM(
        __ubuf__ float *__restrict__ dst,
        __ubuf__ float *__restrict__ src,
        __ubuf__ float *__restrict__ src1,
        uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n)
    {
        pipe_barrier(PIPE_V);
        for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
            vdiv((dst + (vadd_idx * FLOAT_VECTOR_SIZE)), (src + (vadd_idx * FLOAT_VECTOR_SIZE)), (src1),
                sub_m,
                1,
                1,
                0,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                1
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            vdiv((dst + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (src + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (src1),
                sub_m,
                1,
                1,
                0,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                1
            );
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ __attribute__((always_inline)) inline void TensorMulRepeatM(
        __ubuf__ float *__restrict__ dst,
        __ubuf__ float *__restrict__ src,
        __ubuf__ float *__restrict__ src1,
        uint32_t sub_m, uint32_t qk_n, uint32_t qk_round_n, uint32_t src1BlockStride
    ) {
        pipe_barrier(PIPE_V);
        for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
            vmul((dst + (vadd_idx * FLOAT_VECTOR_SIZE)), (src + (vadd_idx * FLOAT_VECTOR_SIZE)), (src1),
                sub_m,
                1,
                1,
                src1BlockStride,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                1
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            vmul((dst + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (src + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (src1),
                sub_m,
                1,
                1,
                src1BlockStride,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE,
                1
            );
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }
        pipe_barrier(PIPE_V);
    }


   __aicore__ __attribute__((always_inline)) inline void SoftmaxStage1(
        __gm__ IN_DTYPE *__restrict__ p_gm_tensor,
        __gm__ mm1CopyType *__restrict__ s_gm_tensor,
        __gm__ OUT_DTYPE *__restrict__ mask_gm_tensor,
        __ubuf__ float *__restrict__ dm32_ubuf_tensor,
        __ubuf__ float *__restrict__ ll_ubuf_tensor,
        __ubuf__ float *__restrict__ pm32_ubuf_tensor,
        uint32_t n_idx,
        uint32_t qk_n,
        uint32_t qk_round_n,
        uint32_t sub_m,
        uint32_t mask_offset,
        const uint32_t sub_n_loop,
        const uint32_t cur_batch,
        const uint32_t start_kv,
        const uint32_t real_n_loop,
	    const uint32_t head_idx,
        const uint32_t pm_flag_scalar
    )
    {
        uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
        float quantMax = (float)1 / (float)127;
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            copy_gm_to_ubuf(
                ls32_ubuf_tensor,
                s_gm_tensor,
                0,                        // sid
                1,                        // nBurst
                sub_m * qk_round_n / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                        // srcGap
                0                         // dstGap
            );


        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE; ++vadd_idx) {
            vmuls((ls32_ubuf_tensor + (vadd_idx * FLOAT_VECTOR_SIZE)), (ls32_ubuf_tensor + (vadd_idx * FLOAT_VECTOR_SIZE)), tor,
                sub_m,
                1,
                1,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE
            );
        }
        if (qk_n % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(qk_n % FLOAT_VECTOR_SIZE);
            vmuls((ls32_ubuf_tensor + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (ls32_ubuf_tensor + (qk_n / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), tor,
                sub_m,
                1,
                1,
                qk_round_n / FLOAT_BLOCK_SIZE,
                qk_round_n / FLOAT_BLOCK_SIZE
            );
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }
        pipe_barrier(PIPE_V);

        if (max_context_len != 0) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            AddMask(mask_gm_tensor, mask_ubuf_tensor, mask32_ubuf_tensor, sub_m, qk_n, qk_round_n, mask_offset);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }

        // *** lm = rowmax(ls)
        ReduceMaxRepeatM(lm32_ubuf_tensor, ls32_ubuf_tensor, lp32_ubuf_tensor, sub_m, qk_n, qk_round_n);
        if (n_idx != 0) {
            // *** hm = vmax(lm, gm)
            vmax((hm32_ubuf_tensor), (lm32_ubuf_tensor), (gm32_ubuf_tensor),
                sub_m_d64,
                1,
                1,
                1,
                8,
                8,
                8
            );
            pipe_barrier(PIPE_V);
            // *** dm = gm - hm
            vsub((dm32_ubuf_tensor), (gm32_ubuf_tensor), (hm32_ubuf_tensor),
                sub_m_d64,
                1,
                1,
                1,
                8,
                8,
                8
            );
            pipe_barrier(PIPE_V);
        } else {
            // *** hm = lm
            copy_ubuf_to_ubuf(
                hm32_ubuf_tensor,
                lm32_ubuf_tensor,
                0,                         // sid
                1,                         // nBurst
                round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                         // srcGap
                0                          // dstGap
            );
            pipe_barrier(PIPE_V);
        }
        // *** gm = hm
        copy_ubuf_to_ubuf(
            gm32_ubuf_tensor,
            hm32_ubuf_tensor,
            0,                         // sid
            1,                         // nBurst
            round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
            0,                         // srcGap
            0                          // dstGap
        );
        pipe_barrier(PIPE_V);
        // *** hm_block = expand_to_block(hm), materialized in tv
        if constexpr (SplitKV) {
            if (n_idx == 0) {
                if (gl_flag_scalar == 1) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                    gl_flag_scalar = 0;
                }
            }
        }
        // *** ls = ls - hm_block
        TensorSubValueRepeatM(ls32_ubuf_tensor, ls32_ubuf_tensor,
                           hm32_ubuf_tensor, tv32_ubuf_tensor,
                           sub_m, round_sub_m, qk_n, qk_round_n);
        // *** ls = exp(ls)
        vexp((ls32_ubuf_tensor), (ls32_ubuf_tensor),
                (sub_m * qk_round_n + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                8,
                8
            );
        pipe_barrier(PIPE_V);
            // *** lp = castfp32to16(ls)
            conv_v<float, OUT_DTYPE>(lp_ubuf_tensor,
                ls32_ubuf_tensor,
                (sub_m * qk_round_n + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
                1,                               // dstBlockStride
                1,                               // srcBlockStride
                4,                               // dstRepeatStride
                8                                // srcRepeatStride
            );
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm(
            p_gm_tensor,
            lp_ubuf_tensor,
            0,                        // sid
            1,                        // nBurst
            sub_m * qk_round_n * T_BLOCK_OFFSET / T_BLOCK_SIZE,  // lenBurst
            0,                        // srcGap
            0                         // dstGap
        );
        if (max_context_len != 0){
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        // *** ll = rowsum(ls32)
        ReduceSumRepeatM(ll_ubuf_tensor, ls32_ubuf_tensor, sub_m, qk_n, qk_round_n);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        pipe_barrier(PIPE_V);
    }


    __aicore__ __attribute__((always_inline)) inline void SoftmaxStage2(
        __gm__ mm2CopyType *__restrict__ o_tmp_gm_ptr,
        __ubuf__ float *__restrict__ dm32_ubuf_tensor,
        __ubuf__ float *__restrict__ ll_ubuf_tensor,
        __ubuf__ float *__restrict__ pm32_ubuf_tensor,
        uint32_t n_idx,
        uint32_t n_loop,
        uint32_t qk_n,
        uint32_t qk_round_n,
        uint32_t sub_m,
        uint64_t l_offset,
        uint64_t o_offset,
        uint32_t head_idx,
        uint32_t pm_flag_scalar)
    {
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 64
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;
        uint32_t round_k =  RoundUp<16>(__k);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_ubuf(
                lo_ubuf_tensor,
                o_tmp_gm_ptr,
                0,                    // sid
                1,                    // nBurst
                sub_m * round_k / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                    // srcGap
                0                     // dstGap
            );
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        // *** Update L and O
        if (n_idx != 0) {
            // *** dm = exp(dm)
            vexp((dm32_ubuf_tensor), (dm32_ubuf_tensor),
                sub_m_d64,
                1,
                1,
                8,
                8
            );
            pipe_barrier(PIPE_V);
            // *** gl = dm * gl
            vmul((gl32_ubuf_tensor), (dm32_ubuf_tensor), (gl32_ubuf_tensor),
                sub_m_d64,
                1,
                1,
                1,
                8,
                8,
                8
            );
            pipe_barrier(PIPE_V);
            // *** gl = ll + gl
            vadd((gl32_ubuf_tensor), (gl32_ubuf_tensor), (ll_ubuf_tensor),
                sub_m_d64,
                1,
                1,
                1,
                8,
                8,
                8
            );
            pipe_barrier(PIPE_V);
            // *** dm_block = expand_to_block(dm), materialized in tv
            vbrcb((__ubuf__ uint32_t *)tv32_ubuf_tensor, (__ubuf__ uint32_t *)dm32_ubuf_tensor,
            1,
            8,
            round_sub_m / FLOAT_BLOCK_SIZE
        );
            pipe_barrier(PIPE_V);
            if (go_flag_scalar == 1) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                go_flag_scalar = 0;
            }
            // *** go = go * dm_block
            for (uint32_t vmul_idx = 0; vmul_idx < __k / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                vmul((go32_ubuf_tensor + (vmul_idx * FLOAT_VECTOR_SIZE)), (go32_ubuf_tensor + (vmul_idx * FLOAT_VECTOR_SIZE)), (tv32_ubuf_tensor),
                sub_m,
                1,
                1,
                0,
                round_k / FLOAT_BLOCK_SIZE,
                round_k / FLOAT_BLOCK_SIZE,
                1
            );
            }
            if (__k % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__k % FLOAT_VECTOR_SIZE);
                vmul((go32_ubuf_tensor + (__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (go32_ubuf_tensor + (__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (tv32_ubuf_tensor),
                sub_m,
                1,
                1,
                0,
                round_k / FLOAT_BLOCK_SIZE,
                round_k / FLOAT_BLOCK_SIZE,
                1
            );
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }
            pipe_barrier(PIPE_V);
            // *** go = lo + go
            vadd((go32_ubuf_tensor), (go32_ubuf_tensor), (lo_ubuf_tensor),
                (sub_m * round_k + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                1,
                1,
                1,
                8,
                8,
                8
            );
            pipe_barrier(PIPE_V);
        } else {
            // *** gl = ll
            copy_ubuf_to_ubuf(
                gl32_ubuf_tensor,
                ll_ubuf_tensor,
                0,                // sid
                1,                // nBurst
                round_sub_m / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                // srcGap
                0                 // dstGap
            );
            pipe_barrier(PIPE_V);
            if (go_flag_scalar == 1) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                go_flag_scalar = 0;
            }
            // *** go = lo
            copy_ubuf_to_ubuf(
                go32_ubuf_tensor,
                lo_ubuf_tensor,
                0,                    // sid
                1,                    // nBurst
                sub_m * round_k / FLOAT_BLOCK_SIZE,  // lenBurst
                0,                    // srcGap
                0                     // dstGap
            );
            pipe_barrier(PIPE_V);
        }

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        if (n_idx == n_loop - 1) {
            // *** gl_block = expand_to_block(gl), materialized in tv
            vbrcb((__ubuf__ uint32_t *)tv32_ubuf_tensor, (__ubuf__ uint32_t *)gl32_ubuf_tensor,
            1,
            8,
            round_sub_m / FLOAT_BLOCK_SIZE
        );
            pipe_barrier(PIPE_V);
            // *** go = go / gl_block
            for (uint32_t vdiv_idx = 0; vdiv_idx < __k / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                vdiv((go32_ubuf_tensor + (vdiv_idx * FLOAT_VECTOR_SIZE)), (go32_ubuf_tensor + (vdiv_idx * FLOAT_VECTOR_SIZE)), (tv32_ubuf_tensor),
                sub_m,
                1,
                1,
                0,
                round_k / FLOAT_BLOCK_SIZE,
                round_k / FLOAT_BLOCK_SIZE,
                1
            );
            }
            if (__k % FLOAT_VECTOR_SIZE > 0) {
                __set_mask(__k % FLOAT_VECTOR_SIZE);
                vdiv((go32_ubuf_tensor + (__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (go32_ubuf_tensor + (__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE)), (tv32_ubuf_tensor),
                sub_m,
                1,
                1,
                0,
                round_k / FLOAT_BLOCK_SIZE,
                round_k / FLOAT_BLOCK_SIZE,
                1
            );
                set_vector_mask((uint64_t)-1, (uint64_t)-1);  // fix hidden_size=96
            }
            pipe_barrier(PIPE_V);

            if constexpr (SplitKV) {
                // log(l)
                    vln((tv32_ubuf_tensor), (tv32_ubuf_tensor),
                sub_m,
                1,
                1,
                8,
                8
            );
                    pipe_barrier(PIPE_V);
                    vbrcb((__ubuf__ uint32_t *)hm32_ubuf_tensor, (__ubuf__ uint32_t *)gm32_ubuf_tensor,
            1,
            8,
            round_sub_m / FLOAT_BLOCK_SIZE
        );
                    pipe_barrier(PIPE_V);
                    // logf(lse_sum) + lse_max
                    vadd((tv32_ubuf_tensor), (tv32_ubuf_tensor), (hm32_ubuf_tensor),
                sub_m,
                1,
                1,
                1,
                8,
                8,
                8
            );
                    CopyScale(sub_m, l_offset, o_offset);
            } else {

                // *** go = castfp32to16(go)
                conv_v<float, OUT_DTYPE>(go_ubuf_tensor,
                    go32_ubuf_tensor,
                    (sub_m * round_k + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
                    1,                            // dstBlockStride
                    1,                            // srcBlockStride
                    4,                            // dstRepeatStride
                    8                             // srcRepeatStride
                );
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                copy_ubuf_to_gm_align_b16(
                    o_gm + ((int64_t)o_offset),
                    go_ubuf_tensor,
                    0,        // sid
                    sub_m,    // nBurst
                    __k * 2,  // lenBurst
                    0,        // leftPaddingNum
                    0,        // rightPaddingNum
                    0,        // srcGap
                    0         // dstGap
                );
            }
            // ********************* move O to GM ************************
            if (go_flag_scalar == 0) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                go_flag_scalar = 1;
            }
        }
    }


    __aicore__ __attribute__((always_inline)) inline void InnerRunVector(uint32_t cur_batch, uint32_t start_head, uint32_t cur_nIndx, uint32_t cur_kv_seqlen, uint32_t cur_head_num,
                                                                         uint32_t offset_tiling, uint32_t kv_seqlen, uint32_t embed_split_size_v, uint32_t embed_split_loop_v)
    {
        uint32_t kv_start_head = start_head / group_num; //30 ~ 32
        uint32_t kv_end_head = (start_head + cur_head_num + group_num - 1) / group_num;
        uint32_t cur_kvhead_num = kv_end_head - kv_start_head;
        uint32_t kv_head_idx = kv_start_head + sub_block_idx * cur_kvhead_num / 2;
        uint32_t head_idx = start_head + sub_block_idx * cur_head_num / 2;
        uint32_t addr_o_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 6 + offset_tiling));
        uint32_t addr_o_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 7 + offset_tiling));
        uint64_t addr_o_scalar = (uint64_t)(((uint64_t)addr_o_high32) << 32 | addr_o_loww32);
        uint32_t mask_high32 = (uint32_t)(*((__gm__ int32_t *)tiling_gm + 10 + offset_tiling));
        uint32_t mask_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling_gm + 14 + offset_tiling));
        uint64_t mask_scalar = (uint64_t)(((uint64_t)mask_high32) << 32 | mask_loww32);
        uint32_t addr_l_high32 = 0;
        uint32_t addr_l_loww32 = 0;
        uint64_t addr_l_scalar = 0;
        uint64_t o_offset = 0;
        uint32_t l_offset = 0;
        // o #((num_tokens, num_heads, kvsplit, head_size))
        // l  (numt_tokens, num_heads, kvsplit)
        if constexpr (SplitKV) {
            addr_l_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 11 + offset_tiling));
            addr_l_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 12 + offset_tiling));
            addr_l_scalar = (uint64_t)(((uint64_t)addr_l_high32) << 32 | addr_l_loww32);
            uint32_t addr_o_fd_high32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 15 + offset_tiling));
            uint32_t addr_o_fd_loww32 = (uint32_t)(*((__gm__ uint32_t *)tiling_gm + 16 + offset_tiling));
            uint64_t addr_o_fd_scalar = (uint64_t)(((uint64_t)addr_o_fd_high32) << 32 | addr_o_fd_loww32);
            o_offset = addr_o_fd_scalar * kv_split_core_num + head_idx * __k * kv_split_core_num + cur_nIndx * __k;
            l_offset = addr_l_scalar + head_idx * kv_split_core_num + cur_nIndx;
        } else {
                o_offset = addr_o_scalar + head_idx * embedding_size;
        }
        uint32_t pp_n_scalar = block_size_calc;
        uint32_t sub_n_loop = pp_n_scalar / block_size;
        uint32_t real_n_loop = (cur_kv_seqlen + block_size - 1) / block_size;

        uint32_t n_loop = (cur_kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;
        uint64_t mask_offset = cur_batch % modCoef / divCoef * batch_stride + head_idx * head_stride + (uint64_t)cur_nIndx * kv_split_per_core;
        mask_offset += mask_scalar;

        uint32_t qk_n = pp_n_scalar;
        uint32_t qk_round_n = RoundUp<BLOCK_SIZE>(qk_n);

        uint32_t qk_n_2 = pp_n_scalar;
        uint32_t qk_round_n_2 = RoundUp<BLOCK_SIZE>(qk_n);

        uint32_t sub_m = (sub_block_idx == 1) ? (cur_head_num - cur_head_num / 2) : cur_head_num / 2;
        uint32_t sub_m_d128 = (sub_m + 127) / 128;  // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + 63) / 64;     // up aligned to 128
        uint32_t round_sub_m = (sub_m + 15) / 16 * 16;

        uint32_t start_kv = cur_nIndx * kv_split_per_core;


        uint32_t hiddenSizeOffset = kv_head_idx * embedding_size;
        uint32_t gm_scale_hidden_size = kv_head_idx * embedding_size;
        uint32_t hiddenSizeOffset1 = k_bias_flag ? hiddenSizeOffset : 0;
        uint32_t hiddenSizeOffset2 = v_bias_flag ? hiddenSizeOffset : 0;
        uint32_t sub_m_kv = (sub_block_idx == 1) ? (cur_kvhead_num - cur_kvhead_num / 2) : cur_kvhead_num / 2;


        for (uint32_t n_idx = 0; n_idx < n_loop; n_idx+=2) {
            if (n_idx == (n_loop - 1)) {
                qk_n = (cur_kv_seqlen - n_idx * pp_n_scalar);
                qk_round_n = RoundUp<16>(qk_n);
            }
            if ((n_idx + 1) == (n_loop - 1)) {
                qk_n_2 = (cur_kv_seqlen - (n_idx + 1) * pp_n_scalar);
                qk_round_n_2 = RoundUp<16>(qk_n_2);
            }
            wait_flag_dev(QK_READY_DECODER);
            /* ************ softmax1 stage1  ************* */
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
            if (sub_m > 0) {
                // input QK shape (sub_m, qk_round_n)
                SoftmaxStage1(
                    p_gm + ((uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET +
                        (uint64_t)sub_block_idx * cur_head_num / 2 * qk_round_n * T_BLOCK_OFFSET),
                    s_gm + ((int64_t)block_idx * TMP_SIZE_DECODER +
                        (int64_t)sub_block_idx * cur_head_num / 2 * qk_round_n),
                    mask_gm + (mask_offset + (uint64_t)n_idx * pp_n_scalar),
                    dm32_ubuf_tensor, ll_ubuf_tensor, pm32_ubuf_tensor,
                    n_idx, qk_n, qk_round_n, sub_m, mask_offset, sub_n_loop, cur_batch, start_kv, real_n_loop, head_idx, pm_flag_scalar1
                );
               // input QK shape (sub_m, qk_round_n)
            }
            pipe_barrier(PIPE_MTE3);
            DdrBarrierBeforeFfts();
            FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY_DECODER);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
            /* ************ softmax1 stage2  ************* */
            if (n_idx + 1 < n_loop) {
                wait_flag_dev(QK_READY_STAGE2);
                if (sub_m > 0) {
                    SoftmaxStage1(
                        p_gm + ((uint64_t)block_idx * TMP_SIZE * T_BLOCK_OFFSET  +
                            (uint64_t)sub_block_idx * cur_head_num / 2 * qk_round_n_2 * T_BLOCK_OFFSET +
                            TMP_SIZE * T_BLOCK_OFFSET / 2),
                        s_gm + ((int64_t)block_idx * TMP_SIZE_DECODER +
                            (int64_t)sub_block_idx * cur_head_num / 2 * qk_round_n_2 +
                            TMP_SIZE_DECODER / 2),
                        mask_gm + (mask_offset + (uint64_t)(n_idx + 1) * pp_n_scalar),
                        dm32_stage2_ubuf_tensor, ll_stage2_ubuf_tensor, pm32_ubuf_stage2_tensor,
                        (n_idx + 1), qk_n_2, qk_round_n_2, sub_m, mask_offset, sub_n_loop, cur_batch, start_kv, real_n_loop, head_idx, pm_flag_scalar2
                    );

                }
                pipe_barrier(PIPE_MTE3);
                DdrBarrierBeforeFfts();
                FftsCrossCoreSync<PIPE_MTE3, 2>(SOFTMAX_READY_STAGE2);
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
            /* ************ softmax2 stage1  ************* */
            wait_flag_dev(UPDATE_READY_DECODER);
            uint32_t embed_split_size = embed_split_size_v;
            uint32_t round_embed_split_size = RoundUp<BLOCK_SIZE>(embed_split_size);
            if (sub_m > 0) {
                    SoftmaxStage2(
                        o_tmp_gm + ((int64_t)block_idx * TMP_SIZE +
                        sub_block_idx * cur_head_num / 2 * RoundUp<16>(__k)),
                        dm32_ubuf_tensor, ll_ubuf_tensor, pm32_ubuf_tensor,
                        n_idx, n_loop, qk_n, RoundUp<T_BLOCK_SIZE>(qk_round_n), sub_m, l_offset, o_offset, head_idx, pm_flag_scalar1);
            }
            /* ************ softmax2 stage2  ************* */
            embed_split_size = embed_split_size_v;
            round_embed_split_size = RoundUp<BLOCK_SIZE>(embed_split_size);
            if (n_idx + 1 < n_loop) {
                wait_flag_dev(UPDATE_READY_STAGE2);
                if (sub_m > 0) {
                        SoftmaxStage2(
                            o_tmp_gm + ((int64_t)block_idx * TMP_SIZE +
                            sub_block_idx * cur_head_num / 2 * RoundUp<16>(__k) +
                                TMP_SIZE / 2),
                            dm32_stage2_ubuf_tensor, ll_stage2_ubuf_tensor, pm32_ubuf_stage2_tensor,
                            (n_idx + 1), n_loop, qk_n_2, RoundUp<T_BLOCK_SIZE>(qk_round_n_2), sub_m, l_offset, o_offset, head_idx, pm_flag_scalar2);
                }
            }
        }
    }

private:

    __gm__ mm1CopyType *__restrict__ s_gm{nullptr};
    __gm__ IN_DTYPE *__restrict__ p_gm{nullptr};
    __gm__ mm2CopyType *__restrict__ o_tmp_gm{nullptr};
    __gm__ float *__restrict__ go_gm{nullptr};
    __gm__ float *__restrict__ o_core_tmp_gm{nullptr};
    __gm__ float *__restrict__ l_gm{nullptr};
    __gm__ int32_t* __restrict__ gm_block_tables_{nullptr};

    __gm__ OUT_DTYPE *__restrict__ o_gm{nullptr};
    __gm__ OUT_DTYPE *__restrict__ mask_gm{nullptr};
    __gm__ uint8_t *__restrict__ tiling_gm{nullptr};
    __gm__ float *__restrict__ logN_gm{nullptr};

    UbufAlloc<pagedAttnVariant> UbAllocator;
    const uint32_t ls32_ubuf_offset = UbAllocator.ls32_ubuf_offset;
    const uint32_t lp_ubuf_offset = UbAllocator.lp_ubuf_offset;
    const uint32_t lp32_ubuf_offset = UbAllocator.lp32_ubuf_offset;
    const uint32_t mask_ubuf_offset = UbAllocator.mask_ubuf_offset;
    const uint32_t lo_ubuf_offset = UbAllocator.lo_ubuf_offset;
    const uint32_t mask32_ubuf_offset = UbAllocator.mask32_ubuf_offset;
    const uint32_t ls16_ubuf_offset = UbAllocator.ls16_ubuf_offset;

    const uint32_t lm32_ubuf_offset = UbAllocator.lm32_ubuf_offset;
    const uint32_t hm32_ubuf_offset = UbAllocator.hm32_ubuf_offset;
    const uint32_t pm32_ubuf_offset = UbAllocator.pm32_ubuf_offset;
    const uint32_t pm32_ubuf_stage2_offset = UbAllocator.pm32_ubuf_stage2_offset;
    const uint32_t dm32_ubuf_offset = UbAllocator.dm32_ubuf_offset;
    const uint32_t dm32_ubuf_stage2_offset = UbAllocator.dm32_ubuf_stage2_offset;
    const uint32_t ll_ubuf_offset = UbAllocator.ll_ubuf_offset;
    const uint32_t ll_ubuf_stage2_offset = UbAllocator.ll_ubuf_stage2_offset;
    const uint32_t gm32_ubuf_offset = UbAllocator.gm32_ubuf_offset;
    const uint32_t gl_ubuf_offset = UbAllocator.gl_ubuf_offset;
    const uint32_t gl32_ubuf_offset = UbAllocator.gl32_ubuf_offset;
    const uint32_t go_ubuf_offset = UbAllocator.go_ubuf_offset;
    const uint32_t go32_ubuf_offset = UbAllocator.go32_ubuf_offset;
    const uint32_t tv32_ubuf_offset = UbAllocator.tv32_ubuf_offset;





    __ubuf__ float *ls32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)ls32_ubuf_offset);
    __ubuf__ half *ls16_ubuf_tensor = reinterpret_cast<__ubuf__ half *>((uintptr_t)ls32_ubuf_offset);
    __ubuf__ int32_t *lsint32_ubuf_tensor = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)ls32_ubuf_offset);
    __ubuf__ IN_DTYPE *lp_ubuf_tensor = reinterpret_cast<__ubuf__ IN_DTYPE *>((uintptr_t)lp_ubuf_offset);
    __ubuf__ float *lp32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)lp32_ubuf_offset);
    __ubuf__ OUT_DTYPE *mask_ubuf_tensor = reinterpret_cast<__ubuf__ OUT_DTYPE *>((uintptr_t)mask_ubuf_offset);
    __ubuf__ float *lo_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)lo_ubuf_offset);
    __ubuf__ int32_t *loint32_ubuf_tensor = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)lo_ubuf_offset);
    __ubuf__ float *mask32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)mask32_ubuf_offset);
    __ubuf__ float *lm32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)lm32_ubuf_offset);
    __ubuf__ float *hm32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)hm32_ubuf_offset);
    __ubuf__ float *pm32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)pm32_ubuf_offset);
    __ubuf__ float *pm32_ubuf_stage2_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)pm32_ubuf_stage2_offset);
    __ubuf__ float *gm32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)gm32_ubuf_offset);
    __ubuf__ float *dm32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)dm32_ubuf_offset);

    __ubuf__ float *dm32_stage2_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)dm32_ubuf_stage2_offset);
    __ubuf__ float *ll_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)ll_ubuf_offset);
    __ubuf__ float *ll_stage2_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)ll_ubuf_stage2_offset);
    __ubuf__ OUT_DTYPE *gl_ubuf_tensor = reinterpret_cast<__ubuf__ OUT_DTYPE *>((uintptr_t)gl_ubuf_offset);
    __ubuf__ float *gl32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)gl32_ubuf_offset);
    __ubuf__ float *tv32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)tv32_ubuf_offset);
    __ubuf__ OUT_DTYPE *go_ubuf_tensor = reinterpret_cast<__ubuf__ OUT_DTYPE *>((uintptr_t)go_ubuf_offset);
    __ubuf__ float *go32_ubuf_tensor = reinterpret_cast<__ubuf__ float *>((uintptr_t)go32_ubuf_offset);
    __ubuf__ int32_t *goint32_ubuf_tensor = reinterpret_cast<__ubuf__ int32_t *>((uintptr_t)go32_ubuf_offset);


    __gm__ OUT_DTYPE *gm_k16_ping_{nullptr};
    __gm__ OUT_DTYPE *gm_k16_pong_{nullptr};
    __gm__ OUT_DTYPE *gm_v16_ping_{nullptr};
    __gm__ OUT_DTYPE *gm_v16_pong_{nullptr};

    uint32_t k_bias_flag{0};
    uint32_t v_bias_flag{0};

    uint32_t go_flag_scalar{1};
    uint32_t gl_flag_scalar{1};
    uint32_t pm_flag_scalar1{1};
    uint32_t pm_flag_scalar2{0};
    ScaleType scaleType = ScaleType::SCALE_TOR;
    float tor_logN{0};
    uint32_t num_tokens{0};
    uint32_t q_heads{0};
    uint32_t num_kv_heads{0};
    uint32_t embedding_size{0};
    uint32_t embedding_size_v{0};
    uint32_t block_size{0};
    uint32_t max_context_len{0};
    uint32_t start_head{0};
    uint32_t cur_head_num{0};
    uint32_t __k{0};
    uint32_t round_k{0};
    uint32_t __v{0};
    uint32_t round_v{0};
    uint32_t cur_batch{0};
    float tor{0};
    uint64_t sub_block_idx{0};
    uint32_t batch_stride{0};
    uint32_t head_stride{0};
    uint64_t former_batch{0};
    uint32_t former_head_split{0};
    uint32_t split_size{0};
    uint64_t tail_batch{0};
    uint32_t tail_head_split{0};
    uint32_t core_per_batch{0};
    uint32_t process_num{0};
    uint32_t block_idx{0};
    uint32_t block_num{1};
    uint32_t tiling_head_size{0};
    uint32_t tiling_para_size{0};
    uint32_t kv_split_per_core{0};
    uint32_t kv_split_core_num{0};
    uint32_t block_size_calc{0};
    uint32_t former_group_num_move{1};
    uint32_t tail_group_num_move{1};
    uint32_t embed_split_size_v_former{0};
    uint32_t embed_split_loop_v_former{1};
    uint32_t embed_split_size_v_tail{0};
    uint32_t embed_split_loop_v_tail{1};


    uint32_t modCoef{0xffffffff}; // batch_idx % modCoef (multi-head adaptive compression tiling)
    uint32_t divCoef{1}; // batch_idx / divCoef (multi-head adaptive compression tiling)
    uint32_t q_head_original{0};
    uint32_t compressHead{0};

    uint32_t max_num_blocks_per_query{0};
    uint32_t group_num{0};

    uint32_t prefill_batch_size_;
    uint32_t decoder_batch_size_;
};
#endif
