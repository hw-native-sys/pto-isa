# SET_QUANT_SCALAR

## 简介

设置标量量化参数（pre-quantization scale），用于后续 `TPUSH` 操作。标量值根据输出数据类型编码到硬件量化配置寄存器中。

对于 8位输出类型（`int8_t` 或 `uint8_t`），符号位会自动编码到配置值的第 46位。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename OutType, typename... WaitEvents>
PTO_INST RecordEvent SET_QUANT_SCALAR(float preQuantScalar, WaitEvents &...events);
```

## 约束

- `OutType` 必须是支持的量化输出类型（如 `int8_t`、`uint8_t`、`int16_t`、`bfloat16_t`、`half`）。
- `preQuantScalar` 是一个 FP32 标量值，用于指定预量化缩放因子。
- 该指令必须在消费此配置的 `TPUSH` 指令之前调用。
- 标量值编码到硬件寄存器的具体方式属于实现定义行为。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_set_quant_scalar()
{
    float preQuantScale = 0.5f;
    SET_QUANT_SCALAR<int8_t>(preQuantScale);
}
```

## 汇编示例（ASM）

当前公开的汇编参考尚未为 `SET_QUANT_SCALAR` 定义稳定的 PTO-AS 写法。设置量化配置时请使用 C++ intrinsic 形式。
