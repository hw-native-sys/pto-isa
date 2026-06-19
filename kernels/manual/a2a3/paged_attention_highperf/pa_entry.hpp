#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#ifndef PTO_PA_CONTEXT_PARAMS
#define PTO_PA_CONTEXT_PARAMS
#endif
#ifndef PTO_PA_CONTEXT_ARGS
#define PTO_PA_CONTEXT_ARGS
#endif

#include "pa_kernel_impl.hpp"

static __aicore__ __attribute__((always_inline)) void paged_attention_mask_body(
    __gm__ uint8_t *__restrict__ sync,
    uint32_t pto_block_idx,
    uint32_t pto_block_num,
    uint32_t pto_sub_block_id,
    __gm__ uint8_t *__restrict__ q_gm,
    __gm__ uint8_t *__restrict__ k_gm,
    __gm__ uint8_t *__restrict__ v_gm,
    __gm__ uint8_t *__restrict__ block_tables_gm,
    __gm__ uint8_t *__restrict__ mask_gm,
    __gm__ uint8_t *__restrict__ deq_scale1_gm,
    __gm__ uint8_t *__restrict__ offset1_gm,
    __gm__ uint8_t *__restrict__ deq_scale2_gm,
    __gm__ uint8_t *__restrict__ offset2_gm,
    __gm__ uint8_t *__restrict__ razorOffset,
    __gm__ uint8_t *__restrict__ scale_gm,
    __gm__ uint8_t *__restrict__ logN_gm,
    __gm__ uint8_t *__restrict__ eye_gm,
    __gm__ uint8_t *__restrict__ o_gm,
    __gm__ uint8_t *__restrict__ s_gm,
    __gm__ uint8_t *__restrict__ p_gm,
    __gm__ uint8_t *__restrict__ o_tmp_gm,
    __gm__ uint8_t *__restrict__ go_gm,
    __gm__ uint8_t *__restrict__ o_core_tmp_gm,
    __gm__ uint8_t *__restrict__ l_gm,
    __gm__ uint8_t *__restrict__ gm_k16,
    __gm__ uint8_t *__restrict__ gm_v16,
    __gm__ uint8_t *__restrict__ tiling_para_gm)
{
    if (sync != nullptr) {
        set_ffts_base_addr((unsigned long)sync);
    }
    set_atomic_none();
    set_mask_norm();
#ifdef __DAV_C220_VEC__
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
#elif __DAV_C220_CUBE__
    set_padding(0);
    set_nd_para(1ULL);
#endif
    const uint32_t tiling_key_val = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + AtbOps::TILING_KEY_ID));
    uint32_t prefill_batch_size = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + TILING_PREFILL_BS));
    uint32_t decoder_batch_size = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + TILING_DECODER_BS));
    if (tiling_key_val == 0) { // fp16 BN
#ifdef __DAV_C220_CUBE__
        UnpadAttentionDecoderAic<false, TilingKeyType::TILING_HALF_DATA, half, half, half> pa_aic_fp16(prefill_batch_size, decoder_batch_size);
        pa_aic_fp16.SetArgs(sync, q_gm, k_gm, v_gm, block_tables_gm, o_gm, s_gm, p_gm, o_tmp_gm, gm_k16, gm_v16, tiling_para_gm, razorOffset, pto_block_idx, pto_block_num);
        pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
        UnpadAttentionDecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half> pa_aiv(prefill_batch_size, decoder_batch_size);
        pa_aiv.SetArgs(sync, k_gm, v_gm, deq_scale1_gm, offset1_gm, deq_scale2_gm, offset2_gm, block_tables_gm,
            mask_gm, o_gm, s_gm, p_gm, o_tmp_gm, go_gm, o_core_tmp_gm, l_gm, gm_k16, gm_v16, tiling_para_gm, razorOffset, logN_gm, pto_block_idx, pto_block_num, pto_sub_block_id);
        pa_aiv.Run();
#endif
    } else if (tiling_key_val == 1) { // bf16 BN
#ifdef __DAV_C220_CUBE__
        UnpadAttentionDecoderAic<false, TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16> pa_aic_bf16(prefill_batch_size, decoder_batch_size);
        pa_aic_bf16.SetArgs(sync, q_gm, k_gm, v_gm, block_tables_gm, o_gm, s_gm, p_gm, o_tmp_gm, gm_k16, gm_v16, tiling_para_gm, razorOffset, pto_block_idx, pto_block_num);
        pa_aic_bf16.Run();
#elif __DAV_C220_VEC__
        UnpadAttentionDecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16> pa_aiv(prefill_batch_size, decoder_batch_size);
        pa_aiv.SetArgs(sync, k_gm, v_gm, deq_scale1_gm, offset1_gm, deq_scale2_gm, offset2_gm, block_tables_gm,
            mask_gm, o_gm, s_gm, p_gm, o_tmp_gm, go_gm, o_core_tmp_gm, l_gm, gm_k16, gm_v16, tiling_para_gm, razorOffset, logN_gm, pto_block_idx, pto_block_num, pto_sub_block_id);
        pa_aiv.Run();
#endif
    } else if (tiling_key_val == 16) { // fp16 BNS split-kv
#ifdef __DAV_C220_CUBE__
        UnpadAttentionDecoderAic<true, TilingKeyType::TILING_HALF_DATA, half, half, half> pa_aic_fp16(prefill_batch_size, decoder_batch_size);
        pa_aic_fp16.SetArgs(sync, q_gm, k_gm, v_gm, block_tables_gm, o_gm, s_gm, p_gm, o_tmp_gm, gm_k16, gm_v16, tiling_para_gm, razorOffset, pto_block_idx, pto_block_num);
        pa_aic_fp16.Run();
#elif __DAV_C220_VEC__
        UnpadAttentionDecoderAiv<TilingKeyType::TILING_HALF_DATA, half, half, true> pa_aiv(prefill_batch_size, decoder_batch_size);
        pa_aiv.SetArgs(sync, k_gm, v_gm, deq_scale1_gm, offset1_gm, deq_scale2_gm, offset2_gm, block_tables_gm,
            mask_gm, o_gm, s_gm, p_gm, o_tmp_gm, go_gm, o_core_tmp_gm, l_gm, gm_k16, gm_v16, tiling_para_gm, razorOffset, logN_gm, pto_block_idx, pto_block_num, pto_sub_block_id);
        pa_aiv.Run();
#endif
    } else if (tiling_key_val == 17) { // bf16 BNS split-kv
#ifdef __DAV_C220_CUBE__
        UnpadAttentionDecoderAic<true, TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, __bf16> pa_aic_bf16(prefill_batch_size, decoder_batch_size);
        pa_aic_bf16.SetArgs(sync, q_gm, k_gm, v_gm, block_tables_gm, o_gm, s_gm, p_gm, o_tmp_gm, gm_k16, gm_v16, tiling_para_gm, razorOffset, pto_block_idx, pto_block_num);
        pa_aic_bf16.Run();
#elif __DAV_C220_VEC__
        UnpadAttentionDecoderAiv<TilingKeyType::TILING_BF16_DATA, __bf16, __bf16, true> pa_aiv(prefill_batch_size, decoder_batch_size);
        pa_aiv.SetArgs(sync, k_gm, v_gm, deq_scale1_gm, offset1_gm, deq_scale2_gm, offset2_gm, block_tables_gm,
            mask_gm, o_gm, s_gm, p_gm, o_tmp_gm, go_gm, o_core_tmp_gm, l_gm, gm_k16, gm_v16, tiling_para_gm, razorOffset, logN_gm, pto_block_idx, pto_block_num, pto_sub_block_id);
        pa_aiv.Run();
#endif
    }
    pipe_barrier(PIPE_ALL);
}

#ifndef PTO_PA_NO_GLOBAL_ENTRY
extern "C" __global__ __aicore__ void paged_attention_mask(
    __gm__ uint8_t *__restrict__ sync,
    __gm__ uint8_t *__restrict__ q_gm,
    __gm__ uint8_t *__restrict__ k_gm,
    __gm__ uint8_t *__restrict__ v_gm,
    __gm__ uint8_t *__restrict__ block_tables_gm,
    __gm__ uint8_t *__restrict__ mask_gm,
    __gm__ uint8_t *__restrict__ deq_scale1_gm,
    __gm__ uint8_t *__restrict__ offset1_gm,
    __gm__ uint8_t *__restrict__ deq_scale2_gm,
    __gm__ uint8_t *__restrict__ offset2_gm,
    __gm__ uint8_t *__restrict__ razorOffset,
    __gm__ uint8_t *__restrict__ scale_gm,
    __gm__ uint8_t *__restrict__ logN_gm,
    __gm__ uint8_t *__restrict__ eye_gm,
    __gm__ uint8_t *__restrict__ o_gm,
    __gm__ uint8_t *__restrict__ s_gm,
    __gm__ uint8_t *__restrict__ p_gm,
    __gm__ uint8_t *__restrict__ o_tmp_gm,
    __gm__ uint8_t *__restrict__ go_gm,
    __gm__ uint8_t *__restrict__ o_core_tmp_gm,
    __gm__ uint8_t *__restrict__ l_gm,
    __gm__ uint8_t *__restrict__ gm_k16,
    __gm__ uint8_t *__restrict__ gm_v16,
    __gm__ uint8_t *__restrict__ tiling_para_gm)
{
    const uint32_t pto_block_idx = static_cast<uint32_t>(get_block_idx());
    const uint32_t pto_block_num = static_cast<uint32_t>(get_block_num());
#ifdef __DAV_C220_VEC__
    const uint32_t pto_sub_block_id = static_cast<uint32_t>(get_subblockid());
#else
    const uint32_t pto_sub_block_id = 0;
#endif

    paged_attention_mask_body(
        sync,
        pto_block_idx,
        pto_block_num,
        pto_sub_block_id,
        q_gm,
        k_gm,
        v_gm,
        block_tables_gm,
        mask_gm,
        deq_scale1_gm,
        offset1_gm,
        deq_scale2_gm,
        offset2_gm,
        razorOffset,
        scale_gm,
        logN_gm,
        eye_gm,
        o_gm,
        s_gm,
        p_gm,
        o_tmp_gm,
        go_gm,
        o_core_tmp_gm,
        l_gm,
        gm_k16,
        gm_v16,
        tiling_para_gm
    );
}
#endif
