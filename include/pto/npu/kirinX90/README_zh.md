# include/pto/npu/kirinX90/

Kirin X90 系列 PTO 指令实现头文件。

Kirin X90 是昇腾面向消费端场景的 SoC 变体，与 Kirin 9030 共用测试用例。

## 主要文件

```
include/pto/npu/kirinX90/
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
