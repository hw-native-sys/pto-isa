# include/pto/npu/kirin9030/

Kirin 9030 系列 PTO 指令实现头文件。

Kirin 9030 是昇腾面向消费端场景的 SoC 变体，其 PTO 指令实现在部分细节上与其他数据中心 SoC（A2/A3/A5）有所不同。

## 主要文件

```
include/pto/npu/kirin9030/
├── TAdd.hpp
├── TSub.hpp
├── TMul.hpp
├── TDiv.hpp
├── TMatmul.hpp
├── TLoad.hpp
├── TStore.hpp
├── TAssign.hpp
├── TSync.hpp
└── ...                  # 其他指令实现
```

## 相关内容

| 文档 | 内容 |
|------|------|
| [docs/isa/](../../docs/isa/) | ISA 语义与示例 |
| [include/pto/npu/](../README_zh.md) | NPU 实现总入口 |
