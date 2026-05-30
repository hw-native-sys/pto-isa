// Host-side shim compiled into gemm.so.
//
// compile.sh defines KERNEL_CPP as the generated ptoas C++ file. The exported
// call_kernel function is loaded from Python with ctypes and launches Gemm on
// the caller-provided stream.

#ifndef KERNEL_CPP
#error "KERNEL_CPP must be defined at compile time (see compile.sh)."
#endif

#include <cstdint>

#include KERNEL_CPP

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *c, uint8_t *a, uint8_t *b)
{
    Gemm<<<blockDim, nullptr, stream>>>(reinterpret_cast<half *>(c), reinterpret_cast<half *>(a),
                                        reinterpret_cast<half *>(b));
}
