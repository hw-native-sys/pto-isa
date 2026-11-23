# include/pto/npu/kirinX90/

KirinX90 系列 PTO 指令实现头文件。

## 概览

- 按指令（或指令族）组织实现，例如：`TAdd.hpp`、`TMatmul.hpp`、`TLoad.hpp`、`TStore.hpp`
- 包含 KirinX90 专用的算子模式与工具（如适用）

## 相关内容

- ISA 语义与示例：`docs/zh/isa/`
- KirinX90 NPU ST 测试：`tests/npu/Kirin9030/src/st/`，与Kirin9030共用测试用例。
