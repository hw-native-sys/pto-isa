# 比较与选择指令集

比较与选择指令集负责在向量层生成谓词并按谓词选择结果。后续向量和控制操作会依赖这些 mask 语义，因此它们属于架构可见契约。

## 常见操作

- `pto.vcmp`
- `pto.vcmps`
- `pto.vsel`
- `pto.vselr`
- `pto.vselrv2`

## 操作数模型

- `%src0` / `%src1`：向量源操作数
- `%scalar`：标量比较值（用于 `vcmps`）
- `%seed`：输入谓词，限制哪些 lane 参与比较
- `%mask`：选择操作使用的谓词
- `%result`：谓词结果或向量结果

## 比较模式

| 模式 | 含义 |
| --- | --- |
| `eq` | 等于 |
| `ne` | 不等于 |
| `lt` | 小于 |
| `le` | 小于等于 |
| `gt` | 大于 |
| `ge` | 大于等于 |

## 机制

### 比较

```text
vcmp : vector × vector -> mask
vcmps: vector × scalar -> mask
```

### 选择

```text
vsel : mask ? src0 : src1
vselr / vselrv2: 选择语义由具体变体定义
```

## 约束

- 参与比较的向量宽度和类型必须兼容
- `%seed` 或 `%mask` 的宽度必须匹配目标向量宽度
- `vsel` 的两个源向量与结果向量必须共享相同 shape 和 dtype
- `vselr` / `vselrv2` 的变体语义必须由 lowering 精确保留

## 不允许的情形

- 用不匹配的 mask 宽度驱动比较或选择
- 把某个 backend 的隐式谓词来源写成通用 PTO 语义
- 省略 compare mode 对结果的影响

## 相关页面

- [谓词与物化](./predicate-and-materialization_zh.md)
- [向量指令族](README_zh.md)
