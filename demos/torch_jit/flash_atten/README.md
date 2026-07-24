To reproduce the numerical tolerance comparison between the JIT Flash Attention kernel and the reference implementation:

```bash
export PTO_LIB_PATH=...
python fa_compile_and_run.py
```

To instead run benchmark:
```bash
export PTO_LIB_PATH=...
python fa_benchmark.py
```

Which produces a CSV file containing one row per `(sq, sk, kernel)` configuration with the following fields:

- `sq`, `sk` — query and key sequence lengths
- `head_size` — attention head dimension (fixed at 128)
- `kernel` — attention implementation (`npu_fused_attention`, `jit_flash`)
- `time_us` — average execution time in microseconds over 50 iterations
- `tflops` — achieved throughput for the full attention forward pass
- `flops_total` — total operation count used to compute TFLOP/s


## Attention Performance Benchmarks

All benchmarks were run on 910B2 after:

- **10 warm-up iterations**
- **50 timed iterations (average reported)**

The JIT flash kernel parallelizes work across tiles of size 128 along the query dimension.

We define:

- jit_cores = sq / 128

Since larger `sq` launches more parallel JIT cores, raw TFLOP/s naturally increases with `sq`.
To compare performance independent of parallelism, we normalize throughput to a fixed 24-core equivalent for the 910 B2.

The normalized throughput is computed as:
- normalized_jit_tflops = jit_tflops * 24 / jit_cores

---

#### Results (Batch = 1, Head Size = 128)

**Speedup vs Fused Baseline aarch64**

---

| S0 | S1 | Fused µs | JIT µs | Speedup | JIT TFLOP/s | Normalized JIT TFLOP/s |
|----|----|---------:|-------:|--------:|-----------:|----------------------:|
|128|1024|70.338|19.200|3.66×|3.54|84.87|
|128|2048|70.898|23.420|3.03×|5.80|139.15|
|128|4096|72.652|38.317|1.90×|7.09|170.11|
|128|8192|73.365|68.778|1.07×|7.90|189.54|
|256|1024|72.522|18.604|3.90×|7.30|87.59|
|256|2048|73.369|23.939|3.06×|11.34|136.14|
|256|4096|74.567|39.838|1.87×|13.63|163.61|
|256|8192|73.150|71.212|1.03×|15.25|183.06|
|512|1024|71.964|18.603|3.87×|14.60|87.59|
|512|2048|74.180|24.787|2.99×|21.91|131.48|
|512|4096|73.984|41.475|1.78×|26.19|157.15|
|512|8192|75.936|74.903|1.01×|29.01|174.04|
|1024|1024|74.404|19.439|3.83×|27.94|83.82|
|1024|2048|73.102|26.221|2.79×|41.43|124.29|
|1024|4096|75.043|43.127|1.74×|50.38|151.13|
|1024|8192|90.272|76.855|1.17×|56.54|169.62|
|2048|1024|76.777|23.424|3.28×|46.38|69.57|
|2048|2048|75.535|36.055|2.10×|60.26|90.39|
|2048|4096|91.798|59.690|1.54×|72.80|109.20|
|2048|8192|126.845|107.780|1.18×|80.63|120.95|

**Speedup vs Fused Baseline x86_64**

---

| S0 | S1 | Fused µs | JIT µs | Speedup | JIT TFLOP/s | Normalized JIT TFLOP/s |
|----|----|---------:|-------:|--------:|-----------:|----------------------:|
|128|1024|46.83|16.54|2.83×|4.11|98.56|
|128|2048|48.20|23.98|2.01×|5.66|135.96|
|128|4096|47.93|40.94|1.17×|6.64|159.27|
|128|8192|52.58|72.14|0.73×|7.53|180.77|
|256|1024|48.41|16.77|2.89×|8.10|97.20|
|256|2048|50.13|22.96|2.18×|11.83|142.00|
|256|4096|49.19|40.43|1.22×|13.44|161.28|
|256|8192|68.99|72.39|0.95×|15.01|180.15|
|512|1024|52.50|18.03|2.91×|15.07|90.41|
|512|2048|53.62|25.02|2.14×|21.72|130.31|
|512|4096|61.27|41.21|1.49×|26.37|158.23|
|512|8192|88.09|73.93|1.19×|29.40|176.40|
|1024|1024|55.40|18.70|2.96×|29.06|87.17|
|1024|2048|61.38|26.76|2.29×|40.61|121.83|
|1024|4096|84.73|44.20|1.92×|49.17|147.52|
|1024|8192|103.27|77.65|1.33×|55.98|167.95|
|2048|1024|63.49|23.27|2.73×|46.70|70.05|
|2048|2048|84.70|35.15|2.41×|61.83|92.75|
|2048|4096|101.94|58.43|1.74×|74.40|111.59|
|2048|8192|142.38|100.93|1.41×|86.14|129.21|
