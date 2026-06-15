# DaVinci AIC V300 ISA Vector Thread Extension 用户指南摘录

## SMEM_BAR

注意：store 指令可能乱序发射。因此，当多个 store 指令写入同一个 UB 地址时，包括目标地址相同的 SCATTER 指令，程序员需要在这些 store 指令之间插入 MEM_BAR 指令。

### 语法

```text
SMEM_BAR.type
```

---

### 说明

`.type={.VV_ALL .VST_VLD .VLD_VST .VST_VST .VS_ALL .VST_LD .VLD_ST .VST_ST .SV_ALL .ST_VLD .LD_VST .ST_VST}`

该指令用于阻塞 UB 访问。

- `.VV_ALL`
  阻塞 vector load/store 指令执行，直到所有 vector load/store 指令完成。

- `.VST_VLD`
  阻塞 vector load 指令执行，直到所有 vector store 指令完成。

- `.VLD_VST`
  阻塞 vector store 指令执行，直到所有 vector load 指令完成。

- `.VST_VST`
  阻塞 vector store 指令执行，直到所有 vector store 指令完成。

- `.VS_ALL`
  阻塞 scalar load/store 指令执行，直到所有 vector load/store 指令完成。

- `.VST_LD`
  阻塞 scalar load 指令执行，直到所有 vector store 指令完成。

- `.VLD_ST`
  阻塞 scalar store 指令执行，直到所有 vector load 指令完成。

- `.VST_ST`
  阻塞 scalar store 指令执行，直到所有 vector store 指令完成。

- `.SV_ALL`
  阻塞 vector load/store 指令执行，直到所有 scalar load/store 指令完成。

- `.ST_VLD`
  阻塞 vector load 指令执行，直到所有 scalar store 指令完成。

- `.LD_VST`
  阻塞 vector store 指令执行，直到所有 scalar load 指令完成。

- `.ST_VST`
  阻塞 vector store 指令执行，直到所有 scalar store 指令完成。
