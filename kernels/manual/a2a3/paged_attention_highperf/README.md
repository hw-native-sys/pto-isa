# Paged Attention HighPerf PTO-ISA Kernel

Manual PTO-ISA port of the high-performance SPMD paged-attention kernel.

The same source is compiled twice for the mixed AIC/AIV execution model:

- `paged_attention_highperf_aic_kernel`: `dav-c220-cube`
- `paged_attention_highperf_aiv_kernel`: `dav-c220-vec`
