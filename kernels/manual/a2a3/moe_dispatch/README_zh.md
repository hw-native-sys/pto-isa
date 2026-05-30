# MoE Dispatch — PTO-ISA 独立通信算子

## 概述

基于 PTO-ISA 指令集实现的 MegaMoE Dispatch 独立通信算子。
该算子从远端 rank 的共享内存中拉取量化 token，将交织格式的
`[int8 token | float scale]` 行拆分为紧凑的独立输出（`gmA` 和 `gmPerTokenScale`）。

提供三条独立的 kernel 路径：

- **Direct（2步）**：`TLOAD 远端 GM → UB → TSTORE 拆分` — 自适应 UB tiling 的快速路径
- **ViaGM（4步）**：`TGET 远端 GM → 本地 GM → TLOAD → UB → TSTORE 拆分` — 兼容 MegaMoE 的路径
- **WithSync（集成）**：`CrossRankSync → Direct dispatch` — 设备侧路由表计算 + dispatch

## 支持的 AI 处理器

- Ascend A2
- Ascend A3

## 数据流

```
Direct 路径（mode=direct）：
  远端 GM ──TLOAD──▶ UB (ping/pong) ──TSTORE──▶ gmA (token)
                                      ──TSTORE──▶ gmPerTokenScale (scale)

ViaGM 路径（mode=viagm）：
  远端 GM ──TGET──▶ 本地临时 GM ──TLOAD──▶ UB (ping/pong) ──TSTORE──▶ gmA
                                                            ──TSTORE──▶ gmPerTokenScale

WithSync 路径（mode=sync）：
  Phase A：TPE AllGather（TSTORE 远端写 + TWAIT）
      本地 TPE ──TSTORE+DataAsFlag──▶ 所有远端 rank
      ──TWAIT──▶ 接收所有远端 rank 的 TPE

  Phase B：计算路由表（设备侧）
      B.1 去除 DataAsFlag 偏移（向量化 TLOAD/TADDS/TSTORE）
      B.2 计算 cumsumMM 前缀和（向量化 TLOAD/TADD/TSTORE）
      B.3 计算 preSumBeforeRank（标量累加）

  Phase C：使用计算好的路由表调用 MoeDispatchDirect
```

## 算法

```
=== Direct / ViaGM ===
for each 本地专家 (groupIdx):
    for each 远端 rank (dstEpIdx, 按 coreIdx 跨步):
        1. 计算远端 shmem 中的源地址
        2. 计算本地目标偏移 (gmA/gmPerTokenScale)
        3. [Direct] TLOAD 交织行到 UB，TSTORE 拆分 token 和 scale
           [ViaGM]  TGET 行到本地 GM，再 TLOAD→UB→TSTORE 拆分
        4. 事件驱动 ping-pong：TLOAD(N+1) 与 TSTORE(N) 重叠执行
    // 跨 rank 连续流水：rank 之间无气泡

=== WithSync ===
Phase A — TPE AllGather：
    for each 远端 rank i:
        TSTORE 本地 tokenPerExpert 到 rank i 的 TPE 交换区（附加 DataAsFlag 偏移）
    for each 远端 rank i:
        TWAIT 等待 rank i 数据到达（轮询 GM 信号非零）

Phase B — 路由表计算：
    B.1：TLOAD 每行 TPE，TADDS 去除 DataAsFlag 偏移，TSTORE 回写
    SYNCALL（基于 GM 轮询的软件同步）
    B.2：向量化前缀和 — TLOAD row[i]，TADD 累加器，TSTORE → cumsumMM
    B.3：标量循环 — 从 cumsumMM 列累加 preSumBeforeRank[i]

Phase C — Dispatch：
    使用计算好的路由表调用 MoeDispatchDirect
```

## 核心特性

- **三路径设计**：Direct（快速）、ViaGM（兼容）、WithSync（自包含）
- **集成 CrossRankSync**：WithSync 路径在设备侧计算路由表，无需 host 预计算
- **向量化 cumsumMM**：TLOAD/TADD/TSTORE 前缀和，事件流水，32B 对齐 padding
- **软件 SYNCALL**：基于 GM 轮询的跨核同步（避免 FFTS 硬件依赖）
- **自适应 MOVE_NUM**：编译时 `DispatchTraits<TILE_COLS>` 根据 UB 容量自动缩减每批处理行数
- **事件驱动 ping-pong**：`set_flag`/`wait_flag` 重叠 MTE2（TLOAD）和 MTE3（TSTORE）管道
- **跨 rank 连续流水**：ping-pong 状态跨远端 rank 持续，rank 边界无刷新
- **多核并行**：每个 AIV 核处理一个或多个远端 rank（跨步分配）
- **Token/Scale 分离**：远端行 `[int8×K][float scale 对齐到 32B]` → 紧凑 token + scale 输出

## 规格说明

| 项目 | 值 |
|------|---|
| 数据类型（token） | `int8_t` |
| 数据类型（scale） | `float`（存储在 32B 对齐行中） |
| 远端行格式 | `hiddenSize` 字节 token + `UB_ALIGN`（32）字节 padding（scale 在偏移 0 处） |
| 输出 token | `gmA[maxOutputSize, hiddenSize]` — 紧凑，无 padding |
| 输出 scale | `gmPerTokenScale[maxOutputSize]` — 每行 32 字节（float 在偏移 0 处） |
| 默认 hiddenSize | 128 |
| 执行模型 | 仅 AIV（向量核），通过 mpirun 多 rank |

## 目录结构

```
kernels/manual/a2a3/moe_dispatch/
├── moe_dispatch_kernel.cpp     # 设备 kernel：三路径 dispatch
├── main.cpp                    # Host 驱动：MPI 初始化、数据生成、launch、验证
├── moe_dispatch_config.h       # 形状常量、DispatchTraits、workspace 布局
├── hccl_context.h              # 设备侧 HCCL 上下文结构体
├── CMakeLists.txt              # 构建配置（bisheng + dav-c220-vec）
├── run.sh                      # 构建和运行便捷脚本
├── README.md                   # 英文说明
└── README_zh.md                # 本文件
```

## 构建与运行

```bash
# 设置环境
source /mnt/data/ntlab/liulei/set_env_new.sh
export HCCL_WHITELIST_DISABLE=1

# 构建并运行 Direct 路径（默认），2 卡
bash run.sh all --ep 2 --mode direct

# 构建并运行 ViaGM 路径，4 卡
bash run.sh all --ep 4 --mode viagm

# 构建并运行 WithSync 路径（CrossRankSync + Dispatch），2 卡
bash run.sh all --ep 2 --mode sync

# 指定设备（从 4 号卡开始）
bash run.sh all --ep 4 --first-device 4 --mode direct

# 仅构建
bash run.sh build --ep 2 --hidden 128 --debug

# 仅运行（构建后）
bash run.sh run --ep 2 --mode direct

# 清理重新构建
bash run.sh all --ep 4 --mode viagm --clean
```

### run.sh 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ep N` | 2 | rank 数量（EP 数） |
| `--mode direct\|viagm\|sync` | direct | kernel 路径选择 |
| `--first-device N` | 0 | 起始 NPU 设备号 |
| `--hidden N` | 128 | 隐藏层维度（K） |
| `--tokens N` | 64 | 每 rank 最大 token 数 |
| `--max-output N` | 512 | 最大输出行数 |
| `--experts N` | 1 | 每 rank 专家数 |
| `--clean` | — | 强制清理重建 |
| `--debug` | — | 启用调试模式 |

## 与 MegaMoE 的关系

本算子验证了 MegaMoE 的 Dispatch 阶段，可直接作为完整 MegaMoE 融合算子的构建模块：

```
MegaMoE 完整流水线：
  InitRouting → [Dispatch] → GEMM (FFN) → Combine
                 ^^^^^^^^
                 本算子（WithSync 覆盖 InitRouting + Dispatch）
```

- **接口兼容**：参数（`cumsumMM`、`tokenPerExpert`、`preSumBeforeRank`、`shmemBase`）与 MegaMoE 完全对齐
- **WithSync 路径**：等价于 MegaMoE 的 `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2` + `DispatchAndCombine` dispatch 部分
- **ViaGM 路径**：功能上等价于 MegaMoE 的 `DispatchCopyPerToken`
- **Direct 路径**：PTO-ISA 独有优化，绕过中间 GM buffer

## 参考资料

- MegaMoE 源码：`vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp`
- 设计文档：`/mnt/data/ntlab/liulei/docs/megamoe/dispatch_pto_isa_design.md`
- PTO-ISA TGET API：`include/pto/comm/pto_comm_inst.hpp`
