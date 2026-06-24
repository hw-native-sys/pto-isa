## Overview

This example demonstrates how to implement a mixed FA operator using PTO, including project setup, build, and execution.

## Supported AI Processors

- Ascend A2 (Ascend 910B)
- Ascend A3 (Ascend 910C)

## Directory Layout

```
kernels/manual/common/flash_atten/
├── scripts/
│   └── gen_data.py              # Generates input and golden output
├── CMakeLists.txt               # Build configuration
├── fa_performance_kernel.cpp    # Kernel implementation
├── main.cpp                     # Host-side entry point
└── run.sh                       # Convenience script
```

## Build and Run

1. Configure your Ascend CANN environment (example path):

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Run the example:

```bash
cd ${git_clone_path}/kernels/manual/common/flash_atten

# Run default cases (same set baked into generated_cases.*)
bash run.sh -r npu -v Ascend910B1

# Run a single case from the generated set
bash run.sh -r npu -v Ascend910B1 -c case_float_H_128_S0_128_S1_1024

# Provide custom cases (semicolon separated: HEAD_SIZE,S0,S1,CUBE_S0,TILE_S1)
# TILE_S1: supports 128 (=CUBE_S1),256,512
bash run.sh -r npu -v Ascend910B1 --cases "128,128,1024,128,128;128,2048,2048,128,512"

# Provide custom cases and run just one of them
bash run.sh -r npu -v Ascend910B1 --cases "128,128,1024,128,128;128,512,2048,128,128" \
  -c case_float_H_128_S0_128_S1_1024

```

If the run succeeds, the output prints:

```text
test success
```

## Performance

This section records reference performance numbers for the manual Flash Attention kernel in this directory.

Definitions:
- `S0`: query sequence length (rows of Q/O).
- `S1`: key/value sequence length (rows of K/V).
- `Total task time (us)`: end-to-end kernel time per task (microseconds).
- `GOps`: total operations counted for the task.
- `TFLOPS`: `GOps / time`.
- `Normalized TFLOPS`: `TFLOPS × (24 / cores_used)` to estimate full-device throughput on a 24-core A3.

### Summary

- Larger `S1` improves utilization; normalized throughput increases substantially from `S1=1024` to `S1=4096`.
- For `S1=8192`, the best normalized throughput is close across 1–2 cores (≈172.9 vs 171.4), suggesting the kernel is approaching a bandwidth/synchronization limit rather than scaling linearly with core count.
- 4- and 8-core runs show lower normalized throughput, especially at smaller `S1`, which is typically dominated by fixed overheads (pipeline warm-up, synchronization, and memory traffic).
- Simulation numbers can be significantly higher than onboard measurements because the simulator does not model all hardware contention/latency characteristics; use onboard numbers for performance decisions.

### Tables (A2/A3)

Normalized TFLOPS (higher is better):

| Cores | S0 | S1=1024 | S1=2048 | S1=4096 | S1=8192 |
| --- | --- | --- | --- | --- | --- |
| 1 | 128 | 38.27 | 62.62 | 147.08 | 172.86 |
| 2 | 256 | 48.51 | 73.04 | 148.03 | 171.43 |
| 4 | 512 | 38.60 | 58.10 | 138.19 | 149.27 |
| 8 | 1024 | 25.28 | 37.51 | 99.94 | 120.04 |

Total task time (us, lower is better):

| Cores | S0 | S1=1024 | S1=2048 | S1=4096 | S1=8192 |
| --- | --- | --- | --- | --- | --- |
| 1 | 128 | 42.08 | 51.44 | 43.80 | 74.54 |
| 2 | 256 | 33.201 | 44.101 | 43.521 | 75.162 |
| 4 | 512 | 41.721 | 55.441 | 46.621 | 86.322 |
| 8 | 1024 | 63.72 | 85.882 | 64.461 | 107.342 |

GOps:

| Cores | S0 | S1=1024 | S1=2048 | S1=4096 | S1=8192 |
| --- | --- | --- | --- | --- | --- |
| 1 | 128 | 67.11 | 134.22 | 268.44 | 536.87 |
| 2 | 256 | 134.22 | 268.44 | 536.87 | 1073.74 |
| 4 | 512 | 268.44 | 536.87 | 1073.74 | 2147.48 |
| 8 | 1024 | 536.87 | 1073.74 | 2147.48 | 4294.97 |

TFLOPS (measured):

| Cores | S0 | S1=1024 | S1=2048 | S1=4096 | S1=8192 |
| --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1.59 | 2.61 | 6.13 | 7.20 |
| 2 | 256 | 4.04 | 6.09 | 12.34 | 14.29 |
| 4 | 512 | 6.43 | 9.68 | 23.03 | 24.88 |
| 8 | 1024 | 8.43 | 12.50 | 33.31 | 40.01 |

Sim vs NPU comparison (Seq = 2K):

| Run | Total task time (us) | Normalized TFLOPS |
| --- | --- | --- |
| Onboard (NPU) | 147.92 | 21.78 |
| Simulation | 28.59 | 112.66 |

## Operator Description

### Flash Attention Kernel — Overview, Implemenation and Optimization Details

  - **Purpose:** Explain the end-to-end computation performed by the `TFA` kernel (Flash‑Attention 2.0 style tiled/streamed attention), map the maths to the kernel stages (`compute_qk`, `compute_p`, `compute_pv`, `compute_gu`), show tensor shapes between stages for the `tfa` testcases, and provide implementation & tuning notes for each stage.

  ### 1. Computation Flow

  - **FlashAttention 2.0:**

    Let Q ∈ ℝ^{S0×H}, K ∈ ℝ^{H×S1}, V ∈ ℝ^{S1×H} where H is `HEAD_SIZE`.

    The canonical attention product for a single head (without softmax normalization constants) is:

    $$\text{QK} = Q K^\top \in \mathbb{R}^{S0\times S1}$$
    $$P = \operatorname{softmax}\!\left(\frac{\text{QK}}{\tau}\right)\in \mathbb{R}^{S0\times S1}$$
    $$O = P\,V \in \mathbb{R}^{S0\times H}$$

    For Flash Attention the computation of QK and P is split into tile of (S0,S1) and keep updating a running partial sum O as follows:
        

    **Step 1: Local Row Max**
    For each row \(i\), compute the maximum value within the current tile:

    $$
    m_i = \max_j X_{ij}
    $$
    
    **Description:** `local_max` — the maximum of the current tile row.

    **Step 2: Updated Global Max**
    Update the running global maximum per row:

    $$
    M_i = \max\!\big(M_{\text{prev},i},\; m_i\big)
    $$
    **Description:** `new_global_max` — combines prior global max with the local tile max.

    **Step 3: Rescaling Factor for Previous Sums**
    Rescale the previous sums to account for the new global max:

    $$
    	ext{exp\_max}_i = \exp\!\Big(s \cdot (M_{\text{prev},i} - M_i)\Big)
    $$
    **Description:** `l1_exp_max` — exponential factor that rescales old sums when the max increases.

    **Step 4: Per‑Element Exponentials**
    Compute exponentials for each element relative to the new global max:

    $$
    e_{ij} = \exp\!\Big(s \cdot (X_{ij} - M_i)\Big)
    $$
    **Description:** `p_tile_fp32` / `x_exp` — stored per‑element exponentials (FP32 buffer `p_tile_fp32`, with `x_exp` cast to fp16 for matmul).

    **Step 5: Local Sum for This Tile**
    Sum the exponentials across the row for the current tile:

    $$
    \ell_i = \sum_j e_{ij}
    $$
    **Description:** `local_sum` — per‑row sum of exponentials for this tile.

    **Step 6: Updated Global Sum**
    Update the running global sum by combining rescaled previous sum and current local sum:

    $$
    S_i = \text{exp\_max}_i \cdot S_{\text{prev},i} + \ell_i
    $$
    **Description:** `l2_global_sum` — recurrence for numerically stable accumulation.

    **Final Normalized Softmax Output**

    After all tiles are processed, compute the normalized probabilities:

    $$
    p_{ij} = \frac{e_{ij}}{S_i}
    $$

    **Description:** Final softmax probability for element \((i,j)\).

    **Notes**
    - scale $s = \frac{1}{\sqrt{\mathbb{HEAD\_SIZE}}}$
    - The recurrence ensures numerical stability by rescaling prior sums whenever the global max increases.  
    - The kernel stores $e_{ij}$ (`x_exp`) for the PV matmul, while $\text{exp\_max}_i$ and ${S_i}$ are kept for GU accumulation. 

    <!-- Embedded SVG diagram -->
    <div>
    <img src="fa_flows.svg" alt="FA Computation Flowg" />
    </div>

  - **Tensor shape progression:**

    - Inputs:
      - Q: `S0 × HEAD_SIZE` (fp16) 
      - K: `S1 x HEAD_SIZE` (fp16) 
      - V: `S1 × HEAD_SIZE` (fp16)

    - Per-tile intermediate (tile t):
      - qk_tile: `S0 × Cube_S1` (fp32 accumulation) — e.g. `64×128` or `128×128` in added cases
      - p_tile (xexp): `S0 × Cube_S1` (fp16 stored for matmul computation)
      - pv_tile: `S0 × HEAD_SIZE` (fp32) — per-tile partial result

    - Final outputs:
      - O: `S0 × HEAD_SIZE` (fp16/fp32)

  ### 2. Per‑stage implementation & optimizations

  - **compute_qk (cube matmul)**
    - Role: compute Q·K_t for a single S1 tile. Implemented in `compute_qk` (cube pipeline).
    - Implementation notes:
      - Q tile load is optimized for leftTile stationary: when `tile_idx==0` Q is loaded once per cube invocation; subsequent tiles only load K tiles, reduce reloading from global memory.
      - Writes partial qk results either into a per-tile into a compact ping-pong global buffer.
      - make use of general matmul_macro_pto function to carry out computation from matTile to accTile, which determine the CubeK tiling for defining left and right tile, and keep a running state for left and right tile doing ping/pong. 
    - Optimizations: 
     - Use assign_running_acc_tile to allow output accTile doing double buffer between different compute_qk and compute_pv calls
     - `qkPreloadNum` (default 4) also sets `qkp_tile_fifo_size = 1 + qkPreloadNum` to double-buffer qk tiles between the cube producer and vector softmax consumer.

  - **compute_p (TSOFTMAXFA softmax on vector cores)**
    - Role: numerically-stable tiled softmax across S1 produced incrementally by tiles.
    - Implementation notes:
      - Vector tiling: `Vec_S0 = S0 / VEC_CORES` handles per‑subblock rows; vector cores operate on `Vec_S0 × Cube_S1` tiles, `VEC_CORES` determines how S0 rows are split among vector subblocks.
      - Each vector cores uses `get_subblockid()` to index global tensor loading/storing tiles and compute offsets into qk/p/pv/o buffers — this provides natural SPMD parallelism across vector cores.
      - Uses `TSOFTMAXFA` micro-kernel which implements the tiled softmax recurrence:
        The kernel stores per-tile `l1_exp_max` factors and `l2_global_sum` which are later used by GU reduction o_running accumulation, and compute the final O in the last stage.

    - Optimizations & tradeoffs:
      - Use TROWMAX/TROWSUM call with a static tile size to allow most effiecnt implementon for 128/256/512/1024 reduce axis, pls consider to do a TFILLPAD (PAD_MIN/-INF) to convert dynamic valid rows/cols to static for handling dynamic input (e.g. S0 seqlen)
      - Use TROWEXPANDSUB inplace computation (dst==src) to mininize buffer allocation
      - Carefully interleaving 1d reduced tile compute and 2d compute to reuse pipe barrier bubble within vector unit

    - Vector tile UB allocation and reuse (allocate_vec_tile_buffers)
      - Purpose: lay out UB offsets for all per-vector tiles used by the `compute_p`/`compute_gu` path so the vector cores can reuse a small set of UB addresses predictably.
      - Template parameters: `SrcBuffers`, `XexpBuffers`, `pvVecBuffers` and `ExpMaxBuffers` (`ExpMaxBuffers` typically equals `qkp_tile_fifo_size` so each preload buffer has its own `l1_exp_max` tile).
      - Allocation order (reasoning): the helper assigns UB addresses in a fixed sequence so later code can TASSIGN/TSTORE/TLOAD with static offsets:
        - source `qk` vec tiles
        - `m1_local_max` (reduce tile)
        - `m2_global_max` (reduce tile)
        - a single float tile for intermediate exponentials (called `input_reduce_tmp` in the code)
        - `l1_local_sum` (reduce tile)
        - `l2_global_sum` (reduce tile)
        - the array of `l1_exp_max` reduce tiles (size = `ExpMaxBuffers`)
        - the `x_exp` half tiles (size = `XexpBuffers`)
        - finally the `runningOTile` accumulator is placed at the tail (so GU can TLOAD it)

  - **compute_pv (p·V second matmul)**
    - Role: multiply per-tile P (softmax) by the corresponding V tile to produce partial PV accumulation for output heads.
    - Implementation notes:
      - Loads a V tile and the P tile.
      - Writes `pv_tile_fifo` into a global float buffer, into per-tile ping-pong buffers.
    - Optimizations & tradeoffs:
      - `pv_tile_fifo_size` (computed as `1 + qkPreloadNum`) controls PV ping/pong buffering between P production and GU consumption.

  - **compute_gu (reduction / GU stage)**
    - Role: reduce partial pv parts into the final O and perform the final normalization using `l1_exp_max` / `l2_global_sum` where required.
    - Implementation notes:
      - `compute_gu` is vector-core driven and performs per-tile accumulation using `TGU_ND` / `TGU_LAST_ND` macro kernel. The last tile triggers final division by `l2_global_sum`.
    - Optimizations & tradeoffs:
      - Keep `runningOTile` accumulator assigned.
      - Use TROWEXPANDMUL and TROWEXPANDDIV inplace computation (dst==src) to mininize buffer allocation

  ### 3. Pipeline orchestration & cube/vector parallelism 

  - **FA pipeline stages inter-CV FIFOs and intra-stage ping/pong**

    <!-- Embedded SVG diagram -->
    <div>
    <img src="fa_pipeline.svg" alt="Inter CV FIFOs and intra stage ping/pong" />
    </div>

  - **Inter Cube Core and Vector Core pipelines:**
    - The kernel separates cube (matrix) work and vector work into different pipelines to provide computation parallelism.
    - Typical flow in a loop over S1 tiles:
      1. Preload next QK tile(s) on cube pipeline (compute_qk) and P tile(s) on vector pipeline signal with flags.
      2. Vector pipeline (compute_p) waits on qk availability, runs TSOFTMAXFA on that qk chunk, writes p tile and signals pv consumer.
      3. Cube pipeline (compute_pv) consumes p and v to compute pv_tile_fifo (cube matmul style), writes pv to global memory and signals GU consumer.
      4. GU (compute_gu) running on vector cores consumes pv_tile_fifo and accumulates into `runningOTile`.

  - **Intra Cube Core and Vector Core pipelines:**
    - The matmul_macro_pto and assign_running_acc_tile provided the leftTile/rightTile/AccTile double buffering pipeline mechanism for cube core level pipeline accross different perload sequence of compute_qk and compute_pv calls
    - Inputs k_tile, p_tile (intermediate between CV), v_tile matTiles are double buffered to provide smooth pipeline
    - Compute_p qk_tile input and p_tile output are double buffered, and expT has multi-preload-buffer to allow preload result late forwarding

  - **Buffer Allocation and Reuse Strategy**
    - Each stage should have it input and output ping/pong buffer allocated, but the buffer allocation is limited by the hardware on-chip matTile/vecTile buffer (L1/UB) size.
    - Ping‑pong AccTile assignment for accumulators is done via `assign_running_acc_tile()` to avoid write-after-read hazards when overlapping producers/consumers.
    - The design is allow out-of-order execution for Reordering the pipeline stage schedule below.

  - **Reorder the pipeline stage execution schedule to resolve datadpenency**
    - Lets look at an example for Head=128 S0=128 and S1=1024 case, for CUBE_S1=128 tiling, there are totally 4 loops each with compute_qk->compute_p->compute_pv->compute_gu, and there is data depenency between stage in the loop. compute_qk & compute_pv stages are executed in cube core, and compute_pv & compute_gu stages are executed in vector core.


    In theory, without software pipelining (pre-executing qk, p, and later pv) the execution would be fully in sequential below:

    <div>
    <img src="fa_pipeline_preload0_generated.svg" alt="CV Seqential execution" />
    </div>

    With pre-execution of qk, p (and later pv) would resolve the data depenency and keep the vector compute resoruce fully busy, below showing the intra-core (tload,tcompute,tstore) and inter-CV-stage pipeline
    
    QK pre-execution = 2 (Theory)
    
    <div>
    <img src="fa_pipeline_preload2_generated.svg" alt="Pre-execution of QK twice" />
    </div>    

    QK pre-execution = 4 (Theory)
    <div>
    <img src="fa_pipeline_preload4_generated.svg" alt="Pre-execution of QK=4" />
    </div>    

    Actual kernel exection using simulation result for H128 Seqlen=1024, behaviour is similar to the theory one as we excepted.

    Kernel Execution - QK pre-execution = 4 (Sim)
    <div>
    <img src="fa_sim_run_pipeline_h128_s0_128_s1_1024.svg" alt="Pre-execution of QK=4" />
    </div>    

    From the pipeline diagram, we could see the actual bound right now is on TSTORE on the cube side, and the cube utilization is bound ~30%, Next vesion we would see how we could further optiminize this case.

  - **Overlap mechanisms & tuning knobs:**
    - `qkPreloadNum` lets cube pipeline run ahead producing QK tiles for future S1 tiles.
    - `qkp_tile_fifo_size` sets the qk/p FIFO depth (derived as `1 + qkPreloadNum`) for producer/consumer overlap.
    - `pv_tile_fifo_size` mirrors the qk FIFO depth for PV tiles so PV production can overlap with GU.
    - Synchronization uses lightweight device flags to minimize stalls; prefer more asynchronous overlap (larger preload, more buffers) when UB permits.

  ### 4. Multicore task split, tiling and load balancing

  - **Multicore tiling and work distribution:**
    - For BNSD (Batch, no-of-Head, Seqlen, HEAD_SIZE) QKV input, with intermediate QK(S0,S1) during computation, since S1 is the reduce axis, multi-core tiling should be split base on (B,N,S/Cube-S0), while in Flash-decoding case, since (B,N,S/Cube-S0) is small multi-core tiling could split in S1 axis (TODO) and each core keeping partial O and have another kernel to do final GU. The intermediate fifo buffers in case of a very large S0 > Max-no-of-physical-cores (which is 25 in A2/A3) introduce a wastage and uncessary of L2 cache write back it could be optimized base on core id (TODO)

  - **Load balancing guidance:**
    - Consider the compution sparity when casual attention mask applied (TODO), mulit-core tiling also need to take core unbalanced loading along the S0 axis, current appoarch is base on multi-block launchng with block num > physical no of core for large S0. More appoarches we could explorer later.
