# TIMG2COL

## 指令示意图

![TIMG2COL tile operation](../figures/isa/TIMG2COL.svg)

## 简介

用于类卷积工作负载的图像到列变换。

## 数学语义

除非另有说明, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

### IR Level 1（SSA）

```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src,
                            uint16_t posM = 0, uint16_t posK = 0,
                            const Img2colTileConfig<T> &cfg = Img2colTileConfig<T>{}, WaitEvents&... events);
```

## 约束

- This instruction is target/implementation-specific. See `include/pto/npu/*/TImg2col.hpp` for the supported tile types/layouts and config fields.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

