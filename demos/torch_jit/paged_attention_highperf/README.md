# Paged Attention HighPerf Torch JIT Demo

JIT-compiles the PTO-ISA paged-attention highperf kernel and calls it from
PyTorch/NPU tensors through ctypes, following the flash_atten torch_jit pattern.

Set:

    export PTO_LIB_PATH=/mounted_home/pto-isa
    export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0

Run:

    cd /mounted_home/pto-isa/demos/torch_jit/paged_attention_highperf
    python3 pa_compile_and_run.py
    python3 pa_benchmark.py
