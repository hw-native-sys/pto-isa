/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0
*/

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

using namespace PtoTestCommon;

/* ---------- kernel launch declarations (from .so) ---------- */
extern "C" void call_expand_fwd(uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *out, int32_t tokens, int32_t hidden);

/* ---------- bf16 helpers ---------- */
static uint16_t f32_to_bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    return (uint16_t)(u >> 16);
}
static float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* ---------- expand_to_mhc_fwd golden ---------- */
static void golden_expand_fwd(const uint16_t *x, uint16_t *out,
    int tokens, int hidden, int mhc) {
    for (int t = 0; t < tokens; t++)
        for (int m = 0; m < mhc; m++)
            memcpy(out + (t * mhc + m) * hidden, x + t * hidden, hidden * sizeof(uint16_t));
}

int main(int argc, char **argv) {
    const int tokens = 64, hidden = 1280, mhc = 4;
    const int blockDim = 32;

    /* ACL init */
    aclInit(nullptr);
    aclrtSetDevice(0);
    void *stream = nullptr;
    aclrtCreateStream(&stream);

    /* host data */
    const int x_elems = tokens * hidden;
    const int out_elems = tokens * mhc * hidden;
    std::vector<uint16_t> h_x(x_elems), h_out(out_elems, 0), h_golden(out_elems);

    srand(42);
    for (int i = 0; i < x_elems; i++)
        h_x[i] = f32_to_bf16((float)(rand() % 1000 - 500) / 100.0f);

    golden_expand_fwd(h_x.data(), h_golden.data(), tokens, hidden, mhc);

    /* device memory */
    void *d_x = nullptr, *d_out = nullptr;
    aclrtMalloc(&d_x, x_elems * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_out, out_elems * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(d_x, x_elems * 2, h_x.data(), x_elems * 2, ACL_MEMCPY_HOST_TO_DEVICE);

    /* launch */
    call_expand_fwd(blockDim, stream,
        (uint8_t *)d_x, (uint8_t *)d_out, tokens, hidden);
    aclrtSynchronizeStream(stream);

    /* copy back */
    aclrtMemcpy(h_out.data(), out_elems * 2, d_out, out_elems * 2, ACL_MEMCPY_DEVICE_TO_HOST);

    /* verify */
    bool pass = (memcmp(h_out.data(), h_golden.data(), out_elems * 2) == 0);
    printf("expand_to_mhc_fwd: %s\n", pass ? "PASSED" : "FAILED");

    /* cleanup */
    aclrtFree(d_x);
    aclrtFree(d_out);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return pass ? 0 : 1;
}
