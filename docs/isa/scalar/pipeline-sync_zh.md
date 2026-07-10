# 流水线同步

这些 `pto.*` 形式在 PTO 执行阶段之间建立显式 producer-consumer 顺序。即便它们协调的是向量相关流水线，架构上暴露出来的仍然是依赖状态，而不是向量有效载荷算术。

## 同步层次

```text
事件同步 (set_flag / wait_flag)
    ↑
buffer-token 协议 (get_buf / rls_buf)
    ↑
内存屏障 (mem_bar)
    ↑
跨核协调 (set_cross_core / wait_flag_dev / set_intra_block / wait_intra_block)
```

- **事件同步**：基础形式
- **buffer-token**：双缓冲协议，底层仍依赖事件
- **内存屏障**：保证内存可见性，不单独建立跨阶段顺序
- **跨核协调**：profile 相关的更大范围同步

## 覆盖的操作

- `pto.set_flag`
- `pto.wait_flag`
- `pto.pipe_barrier`
- `pto.get_buf`
- `pto.rls_buf`
- `pto.mem_bar`
- `pto.set_cross_core`
- `pto.wait_flag_dev`
- `pto.set_intra_block`
- `pto.wait_intra_block`

## 相关页面

- [控制与配置](./control-and-configuration_zh.md)
- [顺序与同步](../machine-model/ordering-and-synchronization_zh.md)
