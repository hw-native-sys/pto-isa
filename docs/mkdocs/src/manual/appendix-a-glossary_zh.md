# 附录 A. 术语表

**Tile**：片上二维操作数，包含类型、形状、布局与有效区域等元数据。

**GlobalTensor**：对全局内存（GM）的带类型视图，包含形状/步幅等元数据，通常用于 `TLOAD` / `TSTORE`。

**有效区域（valid region）**：Tile 中在某个操作下具有语义定义的元素子集，常写作 `[Rv, Cv]`。

**位置（Location）**：Tile 的存储类别/意图（例如 `Vec`、`Mat`、`Left`、`Right`、`Acc` 等）。

**Block**：并行工作的基本单元，通常由 `block_idx` 标识。

**Sub-block**：Block 内部的细分单元；在适用平台上可由 `subblockid` 标识。

**流水线（Pipeline）**：对 load/transform/compute/store 等阶段的重叠调度，并通过同步机制协调阶段间依赖。

**TSYNC**：用于在阶段类别之间建立顺序关系的同步指令/抽象。
