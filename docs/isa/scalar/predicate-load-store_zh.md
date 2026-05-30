# 谓词加载存储

谓词加载存储指令集在 UB 可见存储与架构谓词指令集之间搬运 `!pto.mask<G>` 状态。谓词是 `pto.v*` 向量操作消费的 lane mask 机制。

## 机制

谓词状态属于标量与控制指令集。`pld*` / `pst*` 操作在 UB 与谓词寄存器之间搬运位图，使谓词可以跨 kernel 持久化或与标量地址计算共享。

### 数据流

```text
Predicate Register File -> UB
UB -> Predicate Register File
```

### 谓词宽度

| 元素类型 | 向量宽度 N | 谓词宽度 |
| --- | :---: | :---: |
| f32 | 64 | 64 bits |
| f16 / bf16 | 128 | 128 bits |
| i8 / u8 | 256 | 256 bits |

### 对齐要求

| 操作 | 对齐要求 |
| --- | --- |
| `plds` / `psts` | 64-bit 对齐 |
| `pld` / `pst` | 基址和偏移都必须满足 64-bit 对齐 |
| `pldi` / `psti` | 立即数偏移必须合法且满足对齐 |
| `pstu` | 不要求首地址对齐，但会维护内部 alignment state |

## 共享约束

- 指针必须位于 UB 地址空间
- 搬运覆盖完整谓词宽度，不支持部分传输
- 与 DMA 或向量计算形成依赖链时，必须显式建立顺序边
- 任一时刻架构上只能有合法的活动谓词状态

## 相关页面

- [控制与配置](./control-and-configuration_zh.md)
- [谓词生成与代数](./predicate-generation-and-algebra_zh.md)
