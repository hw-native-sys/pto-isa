# 性能分析方法

## 带宽估算

```cpp
// RS 阶段数据量
size_t rs_bytes = (nranks - 1) * total_data_bytes / nranks;

// AG 阶段数据量
size_t ag_bytes = (nranks - 1) * total_data_bytes / nranks;

// 实际带宽
float rs_bw_gbps = rs_bytes / (rs_time_us * 1e-6) / 1e9;
float ag_bw_gbps = ag_bytes / (ag_time_us * 1e-6) / 1e9;

// 理论峰值带宽（示例：HCCS 节点内 ~30GB/s per link）
float peak_bw_gbps = 30.0;
float utilization = actual_bw / peak_bw_gbps * 100;
printf("BW utilization: %.1f%%\n", utilization);
```

---

## 性能报告模板

```cpp
void PrintPerfReport(float comp_us, float pipe_us, size_t data_bytes, int nranks)
{
    float comm_est_us = pipe_us - comp_us;
    float speedup = (comp_us + comm_est_us) / pipe_us;
    float overlap_pct = (1.0 - (pipe_us - std::max(comp_us, comm_est_us))
                              / std::min(comp_us, comm_est_us)) * 100;

    size_t rs_bytes = (nranks - 1) * data_bytes / nranks;
    size_t ag_bytes = rs_bytes;
    float rs_bw = rs_bytes / (comm_est_us * 0.5 * 1e-6) / 1e9;
    float ag_bw = ag_bytes / (comm_est_us * 0.5 * 1e-6) / 1e9;

    printf("=== Performance Report ===\n");
    printf("Compute-only:   %.1f us\n", comp_us);
    printf("Pipelined:      %.1f us\n", pipe_us);
    printf("Comm estimate:  %.1f us\n", comm_est_us);
    printf("Speedup:        %.2fx\n", speedup);
    printf("Overlap:        %.1f%%\n", overlap_pct);
    printf("RS bandwidth:   %.1f GB/s\n", rs_bw);
    printf("AG bandwidth:   %.1f GB/s\n", ag_bw);
}
```

---

## Profiling 方法

### Host 侧 Event 计时

```cpp
aclrtEvent start, end;
aclrtCreateEvent(&start);
aclrtCreateEvent(&end);

aclrtRecordEvent(start, computeStream);
launchCompute(..., computeStream);
launchComm(..., commStream);
aclrtRecordEvent(end, computeStream);
aclrtSynchronizeStream(commStream);
aclrtSynchronizeStream(computeStream);

float total_ms;
aclrtEventElapsedTime(&total_ms, start, end);
```

### Compute-only Baseline

```cpp
for (int i = 0; i < COMPUTE_ONLY_ITERS; i++) {
    aclrtRecordEvent(start, computeStream);
    launchComputeOnly(..., computeStream);
    aclrtRecordEvent(end, computeStream);
    aclrtSynchronizeStream(computeStream);
    float ms;
    aclrtEventElapsedTime(&ms, start, end);
    comp_times.push_back(ms);
}
float avg_comp = median(comp_times);
```

### Sequential Baseline

```cpp
aclrtRecordEvent(start, stream);
launchCompute(..., stream);
aclrtSynchronizeStream(stream);
launchComm(..., stream);
aclrtRecordEvent(end, stream);
aclrtSynchronizeStream(stream);
float seq_ms;
aclrtEventElapsedTime(&seq_ms, start, end);
```

---

## 性能迭代策略

```
1. 建立 baseline（compute-only + sequential）
2. 测量 pipelined 性能
3. 计算 speedup 和 overlap%
4. 如果 overlap < 80%：
   a. 检查通信是否太早开始（队列空转）
   b. 检查通信是否太晚开始（Tile 太大）
   c. 检查 Block 负载均衡
5. 如果带宽利用率 < 60%：
   a. 增大 Tile 以减少传输次数
   b. 使用乒乓双缓冲
   c. 检查数据对齐
6. 重复优化迭代
```

---

## msprof 集成

对于更深入的 profiling，可使用 `msprof` 采集硬件 timeline：

```bash
# 采集 kernel 执行 timeline
msprof --output=./prof_data --application="mpirun -np 8 ./my_operator"

# 分析结果
msprof --export=timeline --output=./prof_data
```

`msprof` 可展示各 AICore 的 MTE2/MTE3/Cube/Vec 管道占用率，帮助定位重叠空洞。
