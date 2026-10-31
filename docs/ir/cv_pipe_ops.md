# 核内 CV Pipe 通信操作

本节描述了 PTO ISA 中用于 Cube（AIC）和 Vector（AIV）核心之间 FIFO 风格数据交换的 Pipe 通信接口。这些操作支持 MPMD（多程序多数据）执行模型，允许 Cube 核和 Vector 核通过管道进行异步数据交换。Pipe 条目可以是本地 tile buffer 或 GlobalTensor 风格的全局内存视图描述符。所有 Pipe 通信操作均需通过编译时属性 `id` 进行绑定，该属性将 `initialize_pipe` 与对应的 `tpush`/`tpop`/`tfree` 操作关联。

这一类操作的通用特性包括：

- **dir_mask** 属性：控制 Pipe 通信方向，1 = C2V（Cube 到 Vector），2 = V2C（Vector 到 Cube），3 = 双向
- **id** 属性：编译时整数常量，绑定初始化操作与生产/消费/释放操作
- **slot_size** 属性：字节单位的逻辑 Pipe 总字节大小
- **slot_num** 属性：可选，控制 FIFO 深度（默认 dir_mask=1/2 时为 8，dir_mask=3 时为 4）
- **split** 属性：编译时属性（0=TILE_NO_SPLIT，1=TILE_UP_DOWN，2=TILE_LEFT_RIGHT）
- **tpop 操作返回值**：`tpop_from_aic` 和 `tpop_from_aiv` 是有返回值的操作
- **Pipe 条目类型**：tile entry（`!pto.tile_buf`）或全局条目（`!pto.tensor_view`）

---

## 目录

- [`pto.aic_initialize_pipe` — Cube 侧 Pipe 初始化](#ptoaic_initialize_pipe--cube-侧-pipe-初始化)
- [`pto.aiv_initialize_pipe` — Vector 侧 Pipe 初始化](#ptoaiv_initialize_pipe--vector-侧-pipe-初始化)
- [`pto.tpush_to_aiv` — C2V 生产者推送](#ptotpush_to_aiv--c2v-生产者推送)
- [`pto.tpush_to_aic` — V2C 生产者推送](#ptotpush_to_aic--v2c-生产者推送)
- [`pto.tpop_from_aic` — C2V 消费者弹出](#ptotpop_from_aic--c2v-消费者弹出)
- [`pto.tpop_from_aiv` — V2C 消费者弹出](#ptotpop_from_aiv--v2c-消费者弹出)
- [`pto.tfree_from_aic` — C2V 消费者释放](#ptotfree_from_aic--c2v-消费者释放)
- [`pto.tfree_from_aiv` — V2C 消费者释放](#ptotfree_from_aiv--v2c-消费者释放)
- [`pto.reserve_buffer` — 预留本地消费者缓冲区](#ptoreserve_buffer--预留本地消费者缓冲区)
- [`pto.import_reserved_buffer` — 导入对端预留缓冲区](#ptoimport_reserved_buffer--导入对端预留缓冲区)
- [`pto.section.cube` — Cube 核代码区域](#ptosectioncube--cube-核代码区域)
- [`pto.section.vector` — Vector 核代码区域](#ptosectionvector--vector-核代码区域)
- [`pto.initialize_l2g2l_pipe` — 初始化 L2G2L 管线](#ptoinitialize_l2g2l_pipe--初始化-l2g2l-管线)
- [`pto.initialize_l2l_pipe` — 初始化 L2L 管线](#ptoinitialize_l2l_pipe--初始化-l2l-管线)
- [`pto.talloc_to_aiv` — C2V 生产者 FIFO 分配](#ptotalloc_to_aiv--c2v-生产者-fifo-分配)
- [`pto.talloc_to_aic` — V2C 生产者 FIFO 分配](#ptotalloc_to_aic--v2c-生产者-fifo-分配)

---

## 操作详解

### `pto.aic_initialize_pipe` — Cube 侧 Pipe 初始化

```mlir
pto.aic_initialize_pipe {
                        id = <id>,
                        dir_mask = <dir>,
                        slot_size = <size>,
                        slot_num = <num>,
                        local_slot_num = <local_num>,
                        nosplit = <bool>
                      } (
                        gm_slot_buffer = <buf> : !pto.ptr<T>,
                        gm_slot_tensor = <tensor> : !pto.tensor_view<...>,
                        c2v_consumer_buf = <c2v> : i32,
                        v2c_consumer_buf = <v2c> : i32
                      )
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

在 Cube 核中初始化 Pipe 通道，为双向或单向数据交换建立共享的 FIFO 缓冲区。该操作不产生返回值，仅完成初始化配置。对于本地缓冲区条目，Pipe 管理由运行时处理；对于全局内存条目，初始化绑定 GM FIFO 槽位描述符。

**属性：**

- `id` — Pipe 标识符。编译时整数常量，用于绑定此初始化操作与后续的 `tpush`/`tpop`/`tfree` 操作。必须在同一内核中唯一。

- `dir_mask` — 通信方向掩码。决定 Pipe 的通信模式：
  - `1` — C2V（Cube 到 Vector），仅允许 Cube 端推送，Vector 端消费
  - `2` — V2C（Vector 到 Cube），仅允许 Vector 端推送，Cube 端消费
  - `3` — 双向（Bidirectional），两端均可推送与消费

- `slot_size` — 必需，Pipe 条目的大小（单位：字节）。必须大于 0。

- `slot_num` — 可选，FIFO 深度（FIFO 中的条目数）。默认值：`dir_mask` 为 1 或 2 时为 8，为 3 时为 4。必须大于 0。

- `local_slot_num` — 可选，仅限 Atlas A2系列产品/Atlas A3系列产品。本地 Tile 缓冲区的 FIFO 深度。必须大于 0 且不超过 `slot_num`。Ascend 950PR&950DT系列产品 上必须省略此属性。

- `nosplit` — 可选布尔属性。若为 `true`，禁用 Tile 分割，要求所有绑定的 `tpush`/`tpop`/`tfree` 操作的 `split` 属性为 0（`TILE_NO_SPLIT`）。

- `gm_slot_buffer` — 可选，类型为 `!pto.ptr<T>`。全局内存 FIFO 插槽缓冲区指针。用于 GM 条目的 C2V 或 V2C 生产者推送。

- `gm_slot_tensor` — 可选，类型为 `!pto.tensor_view<...>`。全局内存 FIFO 的张量视图描述符。仅限 GM-only FIFO（全局条目专用）。

- `c2v_consumer_buf` — 可选，类型为 `i32`。C2V 消费者计数缓冲区或导入值。用于同步 Vector 端的消费进度。

- `v2c_consumer_buf` — 可选，类型为 `i32`。V2C 消费者计数缓冲区或导入值。用于同步 Cube 端的消费进度。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 必须出现在 Cube 内核中（`pto.kernel_kind = #pto.kernel_kind<cube>`）。
  - `id` 必须在该内核中唯一。
  - `slot_num` 必须大于 0。
  - 若指定 `local_slot_num`，则必须大于 0 且 `<= slot_num`。
  - `dir_mask` 必须为 1、2 或 3。
  - `slot_size` 必须大于 0。
  - 若 `nosplit` 为 `true`，则所有使用此 `id` 的 `tpush`/`tpop`/`tfree` 操作的 `split` 属性必须为 0。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 必须出现在 Cube 内核中。
  - `local_slot_num` 必须省略（Ascend 950PR&950DT系列产品 不支持）。
  - 其他约束同 Atlas A2系列产品/Atlas A3系列产品。

**示例：**

```mlir
// Atlas A2系列产品/Atlas A3系列产品: 使用 GM 缓冲区和本地槽数配置的 C2V 初始化
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, slot_num = 2, local_slot_num = 1}
    (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
     c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)

// Ascend 950PR&950DT系列产品: 无本地槽配置的 C2V 初始化
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, nosplit = true}
    (c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)
```

---

### `pto.aiv_initialize_pipe` — Vector 侧 Pipe 初始化

```mlir
pto.aiv_initialize_pipe {id = <id>, dir_mask = <dir>, slot_size = <size>,
                         slot_num = <num>, local_slot_num = <local_num>,
                         nosplit = <bool>,
                         gm_slot_buffer = <buf>, gm_slot_tensor = <tensor>,
                         c2v_consumer_buf = <c2v>, v2c_consumer_buf = <v2c>}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

在 Vector 核中初始化 Pipe 通道，与 `pto.aic_initialize_pipe` 结构完全相同。该操作为 Vector 端配置 FIFO 管道，使其能够作为消费者（C2V 模式）或生产者（V2C 模式）参与数据交换。

**属性：**

属性定义与 `pto.aic_initialize_pipe` 完全相同。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品）**
  - 必须出现在 Vector 内核中（`pto.kernel_kind = #pto.kernel_kind<vector>`）。
  - 所有约束同 `pto.aic_initialize_pipe`。

- **实现检查（Ascend 950PR&950DT系列产品）**
  - 必须出现在 Vector 内核中。
  - `local_slot_num` 必须省略。

**示例：**

```mlir
// Atlas A2系列产品/Atlas A3系列产品: Vector 侧 C2V 消费者初始化
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, slot_num = 2, local_slot_num = 1}
    (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
     c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)

// Ascend 950PR&950DT系列产品: Vector 侧初始化
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, nosplit = true}
    (c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)
```

---

### `pto.tpush_to_aiv` — C2V 生产者推送

```mlir
pto.tpush_to_aiv(<pipe_entry> : <pipe_entry_type>)
    {id = <id>, split = <split>}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

从 Cube 核中推送一个 C2V Pipe 条目到 FIFO。对于 tile buffer 条目，执行 tile 传输；对于全局内存条目，提交 GM FIFO 槽位并推进生产者指针。该操作不产生返回值。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `pipe_entry` | `!pto.tile_buf` 或 `!pto.tensor_view` | 要推送的 Pipe 条目，可为 tile buffer 或全局张量视图 |

**返回值：** 无。操作提交条目至 FIFO 后返回。

**属性：**

- `id` — Pipe 标识符。必须与同一 Cube 内核中的 `pto.aic_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式。编译时属性，决定条目如何分割：
  - `0` — `TILE_NO_SPLIT`，不分割
  - `1` — `TILE_UP_DOWN`，按行分割
  - `2` — `TILE_LEFT_RIGHT`，按列分割

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须出现在 Cube 内核中。
  - `id` 必须匹配一个使用 `dir_mask=1` 或 `dir_mask=3` 的 `pto.aic_initialize_pipe` 操作。
  - 对于全局内存条目，操作必须为对应的 `pto.talloc_to_aiv` 所支配（必须在该分配之后）。
  - 若初始化操作的 `nosplit` 为 `true`，则 `split` 必须为 0。

**示例：**

```mlir
// Tile buffer 条目推送（按行分割）
pto.tpush_to_aiv(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
    {id = 0, split = 1}

// 全局内存条目推送（无分割）
pto.tpush_to_aiv(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}
```

---

### `pto.tpush_to_aic` — V2C 生产者推送

```mlir
pto.tpush_to_aic(<pipe_entry> : <pipe_entry_type>)
    {id = <id>, split = <split>}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

从 Vector 核中推送一个 V2C Pipe 条目到 FIFO。结构与 `pto.tpush_to_aiv` 相同，但方向相反（Vector 生产，Cube 消费）。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `pipe_entry` | `!pto.tile_buf` 或 `!pto.tensor_view` | 要推送的 Pipe 条目 |

**返回值：** 无。操作提交条目至 FIFO 后返回。

**属性：**

- `id` — Pipe 标识符。必须与同一 Vector 内核中的 `pto.aiv_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式（定义同 `pto.tpush_to_aiv`）。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须出现在 Vector 内核中。
  - `id` 必须匹配一个使用 `dir_mask=2` 或 `dir_mask=3` 的 `pto.aiv_initialize_pipe` 操作。
  - 其他约束同 `pto.tpush_to_aiv`。

**示例：**

```mlir
// Vector 侧 V2C 推送
pto.tpush_to_aic(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
    {id = 0, split = 1}
```

---

### `pto.tpop_from_aic` — C2V 消费者弹出

```mlir
%entry = pto.tpop_from_aic {id = <id>, split = <split>}
    -> <pipe_entry_type>
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

从 FIFO 中弹出一个 C2V Pipe 条目在 Vector 核中消费。该操作为 SSA 返回值操作，返回一个 tile buffer 或张量视图描述符，后续操作可使用该条目进行数据转移。

**返回值：** 一个 Pipe 条目，类型为 `!pto.tile_buf<...>` 或 `!pto.tensor_view<...>`，取决于初始化配置。

**属性：**

- `id` — Pipe 标识符。必须与同一 Vector 内核中的 `pto.aiv_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式（定义同 `pto.tpush_to_aiv`）。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须出现在 Vector 内核中。
  - `id` 必须匹配一个使用 `dir_mask=1` 或 `dir_mask=3` 的 `pto.aiv_initialize_pipe` 操作。
  - 返回类型必须与初始化配置兼容。

**示例：**

```mlir
// Tile buffer 返回值
%tile = pto.tpop_from_aic {id = 0, split = 1}
    -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>

// 张量视图返回值
%entry = pto.tpop_from_aic {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
```

---

### `pto.tpop_from_aiv` — V2C 消费者弹出

```mlir
%entry = pto.tpop_from_aiv {id = <id>, split = <split>}
    -> <pipe_entry_type>
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

从 FIFO 中弹出一个 V2C Pipe 条目在 Cube 核中消费。结构与 `pto.tpop_from_aic` 相同，但方向相反（Cube 消费来自 Vector 的数据）。

**返回值：** 一个 Pipe 条目，类型为 `!pto.tile_buf<...>` 或 `!pto.tensor_view<...>`。

**属性：**

- `id` — Pipe 标识符。必须与同一 Cube 内核中的 `pto.aic_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须出现在 Cube 内核中。
  - `id` 必须匹配一个使用 `dir_mask=2` 或 `dir_mask=3` 的 `pto.aic_initialize_pipe` 操作。

**示例：**

```mlir
// Cube 侧 V2C 消费
%tile = pto.tpop_from_aiv {id = 0, split = 1}
    -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>
```

---

### `pto.tfree_from_aic` — C2V 消费者释放

```mlir
pto.tfree_from_aic {id = <id>, split = <split>}

// 或（针对全局内存条目）：
pto.tfree_from_aic(<entry> : <pipe_entry_type>)
    {id = <id>, split = <split>}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

释放当前 C2V FIFO 消费者槽位在 Vector 核中。对于 tile buffer 条目，使用无操作数形式；对于全局内存条目，使用包含条目描述符的形式。该操作推进消费者指针，使下一个条目可用。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `entry`（可选）| `!pto.tensor_view<...>` | 针对全局内存条目的释放描述符（仅在释放 GM 条目时需要） |

**返回值：** 无。操作执行释放并返回。

**属性：**

- `id` — Pipe 标识符。必须与同一 Vector 内核中的 `pto.aiv_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须出现在 Vector 内核中。
  - `id` 必须匹配一个使用 `dir_mask=1` 或 `dir_mask=3` 的 `pto.aiv_initialize_pipe` 操作。
  - Tile buffer 释放使用无操作数形式；全局内存释放必须提供条目操作数。

**示例：**

```mlir
// Tile buffer 条目释放（无操作数）
pto.tfree_from_aic {id = 0, split = 1}

// 全局内存条目释放（带条目描述符）
pto.tfree_from_aic(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}
```

---

### `pto.tfree_from_aiv` — V2C 消费者释放

```mlir
pto.tfree_from_aiv {id = <id>, split = <split>}

// 或（针对全局内存条目）：
pto.tfree_from_aiv(<entry> : <pipe_entry_type>)
    {id = <id>, split = <split>}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

释放当前 V2C FIFO 消费者槽位在 Cube 核中。结构与 `pto.tfree_from_aic` 相同，但方向相反。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `entry`（可选）| `!pto.tensor_view<...>` | 针对全局内存条目的释放描述符 |

**返回值：** 无。操作执行释放并返回。

**属性：**

- `id` — Pipe 标识符。必须与同一 Cube 内核中的 `pto.aic_initialize_pipe` 的 `id` 匹配。

- `split` — Tile 分割模式。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须出现在 Cube 内核中。
  - `id` 必须匹配一个使用 `dir_mask=2` 或 `dir_mask=3` 的 `pto.aic_initialize_pipe` 操作。

**示例：**

```mlir
// Cube 侧 V2C 释放（无操作数）
pto.tfree_from_aiv {id = 0, split = 1}

// Cube 侧 V2C 释放（全局内存条目）
pto.tfree_from_aiv(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}
```

---

### `pto.reserve_buffer` — 预留本地消费者缓冲区

```mlir
pto.reserve_buffer {name = <name>, size = <size>,
                    location = <location>, auto = <autoAlloc>} -> i32
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
addr = reserve_local_buffer(name, size, location, autoAlloc)
// 在当前函数中预留一块本地消费者槽位缓冲区，
// 返回其地址供 CV Pipe 初始化或对端导入使用
```

**参数：** 无操作数。

**返回值：** `i32` — 预留缓冲区的地址。

**属性：**

- `name` — 缓冲区名称（字符串），在函数内必须唯一。
- `size` — 缓冲区大小（`i32`），必须大于 0。
- `location` — 地址空间，必须为 `#pto.address_space<vec>` 或 `#pto.address_space<mat>`。
- `autoAlloc` — 是否自动分配（`bool`）。必须为 `true`，基址由编译器在内存规划阶段分配。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须嵌套在 `func.func` 内部。
  - `size` 必须大于 0。
  - `location` 必须为 `vec` 或 `mat`。
  - `name` 在所属函数内必须唯一。

**示例：**

```mlir
%buf = pto.reserve_buffer
    {name = "c2v_slot_buffer", size = 131072,
     location = #pto.address_space<vec>,
     auto = true} -> i32
```

---

### `pto.import_reserved_buffer` — 导入对端预留缓冲区

```mlir
pto.import_reserved_buffer {name = <name>, peer_func = <peer_func>} -> i32
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
addr = import_peer_buffer(name, peer_func)
// 导入由对端函数通过 pto.reserve_buffer 预留的缓冲区地址，
// 用于两侧共享同一块 CV Pipe 相关本地缓冲区
```

**参数：** 无操作数。

**返回值：** `i32` — 对端预留缓冲区的地址。

**属性：**

- `name` — 缓冲区名称（字符串），必须与对端函数中的 `pto.reserve_buffer` 名称匹配。
- `peer_func` — 对端函数引用（`FlatSymbolRefAttr`），必须指向已存在的 `func.func`。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须嵌套在 `func.func` 内部。
  - `peer_func` 引用的函数必须存在。
  - 对端函数中必须存在名称匹配的 `pto.reserve_buffer`。
  - `(name, peer_func)` 组合在所属函数内必须唯一。

**示例：**

```mlir
%import_buf = pto.import_reserved_buffer
    {name = "c2v_slot_buffer",
     peer_func = @vector_kernel} -> i32
```

---

## 完整端到端示例

### C2V：Cube 核生产、Vector 核消费

以下示例展示了一个完整的 C2V Pipe 通信流程：

```mlir
// Cube 核端（C2V 生产者）
func.func @cube_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                       %src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                            v_row=16, v_col=16, blayout=row_major,
                                            slayout=none_box, fractal=1024, pad=0>)
    attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  %c0 = pto.constant 0 : index
  %c1 = pto.constant 1 : index
  %c16 = pto.constant 16 : index

  // 创建全局内存 Pipe 的张量视图
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>

  // 初始化 Pipe（C2V 模式）
  pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  // 为推送分配全局内存条目
  %entry = pto.talloc_to_aiv {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>

  // 获取条目的分区视图并执行数据转移
  %entry_partition = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>

  pto.tstore ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                      v_row=16, v_col=16, blayout=row_major,
                                      slayout=none_box, fractal=1024, pad=0>)
             outs(%entry_partition : !pto.partition_tensor_view<16x16xf32>)

  // 推送条目到 Vector 核
  pto.tpush_to_aiv(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}

  func.return
}

// Vector 核端（C2V 消费者）
func.func @vector_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                         %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                              v_row=16, v_col=16, blayout=row_major,
                                              slayout=none_box, fractal=1024, pad=0>)
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %c0 = pto.constant 0 : index
  %c1 = pto.constant 1 : index
  %c16 = pto.constant 16 : index

  // 创建全局内存 Pipe 的张量视图
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>

  // 初始化 Pipe（C2V 消费者侧）
  pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  // 从 Pipe 弹出条目
  %entry = pto.tpop_from_aic {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>

  // 获取条目的分区视图并执行数据转移
  %entry_partition = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>

  pto.tload ins(%entry_partition : !pto.partition_tensor_view<16x16xf32>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                      v_row=16, v_col=16, blayout=row_major,
                                      slayout=none_box, fractal=1024, pad=0>)

  // 释放消费者槽位
  pto.tfree_from_aic(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}

  func.return
}
```

---

### V2C：Vector 核生产、Cube 核消费

以下示例展示了一个完整的 V2C Pipe 通信流程：

```mlir
  // Vector 核端（V2C 生产者）
  func.func @vector_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                           %src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                                v_row=16, v_col=16, blayout=row_major,
                                                slayout=none_box, fractal=1024, pad=0>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = pto.constant 0 : index
    %c1 = pto.constant 1 : index
    %c16 = pto.constant 16 : index

    %gm_slots = pto.make_tensor_view %gm_slot_buffer,
      shape = [%c16, %c16], strides = [%c16, %c1]
      : !pto.tensor_view<16x16xf32>

    pto.aiv_initialize_pipe {id = 0, dir_mask = 2, slot_size = 1024}
      (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

    %entry = pto.talloc_to_aic {id = 0, split = 0}
      -> !pto.tensor_view<16x16xf32>

    %entry_partition = pto.partition_view %entry,
      offsets = [%c0, %c0], sizes = [%c16, %c16]
      : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>

    pto.tstore ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                        v_row=16, v_col=16, blayout=row_major,
                                        slayout=none_box, fractal=1024, pad=0>)
               outs(%entry_partition : !pto.partition_tensor_view<16x16xf32>)

    pto.tpush_to_aic(%entry : !pto.tensor_view<16x16xf32>)
      {id = 0, split = 0}

    func.return
  }

  // Cube 核端（V2C 消费者）
  func.func @cube_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                         %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                              v_row=16, v_col=16, blayout=row_major,
                                              slayout=none_box, fractal=1024, pad=0>)
      attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0 = pto.constant 0 : index
    %c1 = pto.constant 1 : index
    %c16 = pto.constant 16 : index

    %gm_slots = pto.make_tensor_view %gm_slot_buffer,
      shape = [%c16, %c16], strides = [%c16, %c1]
      : !pto.tensor_view<16x16xf32>

    pto.aic_initialize_pipe {id = 0, dir_mask = 2, slot_size = 1024}
      (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

    %entry = pto.tpop_from_aiv {id = 0, split = 0}
      -> !pto.tensor_view<16x16xf32>

    %entry_partition = pto.partition_view %entry,
      offsets = [%c0, %c0], sizes = [%c16, %c16]
      : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>

    pto.tload ins(%entry_partition : !pto.partition_tensor_view<16x16xf32>)
              outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                        v_row=16, v_col=16, blayout=row_major,
                                        slayout=none_box, fractal=1024, pad=0>)

    pto.tfree_from_aiv(%entry : !pto.tensor_view<16x16xf32>)
      {id = 0, split = 0}

    func.return
  }
```

---

### 本格 FIFO 形：`reserve_buffer` / `import_reserved_buffer`（仅 Ascend 950PR&950DT系列产品）

以下示例展示生产侧用 `import_reserved_buffer` 绑定消费侧 `reserve_buffer` 预留的同名 FIFO：

```mlir
  // 生产侧（Cube）：用 import_reserved_buffer 绑定对端预留的 FIFO
  func.func @cube_kernel() attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0_i32 = arith.constant 0 : i32
    %c2v_import = pto.import_reserved_buffer {
      name = "c2v_fifo",
      peer_func = @vector_kernel
    } -> i32

    pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
      (c2v_consumer_buf = %c2v_import : i32,
       v2c_consumer_buf = %c0_i32 : i32)

    %tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=i32, rows=16, cols=16,
        v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    pto.tpush_to_aiv(%tile : !pto.tile_buf<loc=acc, dtype=i32, rows=16, cols=16,
        v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) {id = 0, split = 0}
    return
  }

  // 消费侧（Vector）：用 reserve_buffer 预留本格 FIFO
  func.func @vector_kernel() attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0_i32 = arith.constant 0 : i32
    %c2v_local = pto.reserve_buffer {
      name = "c2v_fifo",
      size = 4096,
      location = #pto.address_space<vec>,
      auto = true
    } -> i32

    pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
      (c2v_consumer_buf = %c2v_local : i32,
       v2c_consumer_buf = %c0_i32 : i32)

    %recv = pto.tpop_from_aic {id = 0, split = 0}
      -> !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
           v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tfree_from_aic {id = 0, split = 0}
    return
  }
```

---

### `pto.section.cube` — Cube 核代码区域

```mlir
pto.section.cube {
  <body>
}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
#if defined(MACRO_CUBE)
  <body>
#endif
// 降低为条件编译宏保护的代码区域，仅在 Cube 核上编译和执行
```

**参数：** 无。包含一个单块区域（body）。

**返回值：** 无。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 区域为单块、无终止符。
  - body 中的操作仅在 Cube 核（AIC）上下文中有效。

**示例：**

```mlir
pto.section.cube {
  %tile = pto.alloc_tile
      : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
                      v_row=16, v_col=256, blayout=col_major,
                      slayout=row_major, fractal=1024, pad=0>
  pto.tmatmul ins(%a, %b : ...) outs(%tile : ...)
}
```

---

### `pto.section.vector` — Vector 核代码区域

```mlir
pto.section.vector {
  <body>
}
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
#if defined(MACRO_VECTOR)
  <body>
#endif
// 降低为条件编译宏保护的代码区域，仅在 Vector 核上编译和执行
```

**参数：** 无。包含一个单块区域（body）。

**返回值：** 无。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 区域为单块、无终止符。
  - body 中的操作仅在 Vector 核（AIV）上下文中有效。

**示例：**

```mlir
pto.section.vector {
  pto.tload ins(%view : !pto.partition_tensor_view<16x16xf16>)
            outs(%dst : !pto.tile_buf<...>)
  pto.tadd ins(%a, %b : ...) outs(%c : ...)
}
```

---

### `pto.initialize_l2g2l_pipe` — 初始化 L2G2L 管线

```mlir
pto.initialize_l2g2l_pipe {dir_mask = <N>, slot_size = <S>, slot_num = <M>
                           (, local_slot_num = <L>)? (, flag_base = <F>)?
                           (, nosplit = <B>)?}
                          (<gm_addr> : <gm_type> (, <local_addr> : <l_type>)?
                           (, <peer_local_addr> : <p_type>)?)
                          -> !pto.pipe
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
pipe = init_l2g2l_pipe(dir_mask, slot_size, slot_num, gm_addr, ...)
// 初始化一个 Local→Global→Local 管线句柄，用于通过 GM 中转的跨核数据传输
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `gm_addr` | `!pto.ptr<...>` | GM 空间中的中转缓冲区地址 |
| `local_addr` | 可选，本地缓冲区地址 | 本地 DMA 缓冲区（优化路径） |
| `peer_local_addr` | 可选，对端本地缓冲区地址 | 对端核的本地缓冲区 |

**返回值：** `!pto.pipe` — 管线句柄。

**属性：**

- `dir_mask` — 方向掩码（`i8`）。指定 C2V/V2C 方向。
- `slot_size` — 每个 FIFO 槽位大小（`i32`，字节）。
- `slot_num` — FIFO 槽位数量（`i32`）。
- `local_slot_num` — 可选，本地槽位数量（`i32`）。
- `flag_base` — 可选，同步 flag 基地址（`i32`）。
- `nosplit` — 可选，禁用 split 模式（`bool`）。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `slot_size` 和 `slot_num` 必须大于 0。
  - `gm_addr` 必须为合法的 GM 空间类型。

**示例：**

```mlir
%pipe = pto.initialize_l2g2l_pipe
    {dir_mask = 1, slot_size = 16384, slot_num = 2}
    (%gm_ptr : !pto.ptr<f32>) -> !pto.pipe
```

---

### `pto.initialize_l2l_pipe` — 初始化 L2L 管线

```mlir
pto.initialize_l2l_pipe {dir_mask = <N>, slot_size = <S>, slot_num = <M>
                         (, flag_base = <F>)? (, nosplit = <B>)?}
                        (<local_addr> : <l_type>
                         (, <peer_local_addr> : <p_type>)?)
                        -> !pto.pipe
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
pipe = init_l2l_pipe(dir_mask, slot_size, slot_num, local_addr, ...)
// 初始化一个 Local→Local 管线句柄，用于直接通过本地内存的跨核数据传输（无 GM 中转）
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `local_addr` | 本地缓冲区地址 | 本地 DMA 缓冲区 |
| `peer_local_addr` | 可选，对端本地缓冲区地址 | 对端核的本地缓冲区 |

**返回值：** `!pto.pipe` — 管线句柄。

**属性：**

- `dir_mask` — 方向掩码（`i8`）。
- `slot_size` — 每个 FIFO 槽位大小（`i32`，字节）。
- `slot_num` — FIFO 槽位数量（`i32`）。
- `flag_base` — 可选，同步 flag 基地址（`i32`）。
- `nosplit` — 可选，禁用 split 模式（`bool`）。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - `slot_size` 和 `slot_num` 必须大于 0。
  - `local_addr` 必须为合法的本地空间类型。

**示例：**

```mlir
%pipe = pto.initialize_l2l_pipe
    {dir_mask = 1, slot_size = 8192, slot_num = 2, flag_base = 0}
    (%local_buf : i32) -> !pto.pipe
```

---

### `pto.talloc_to_aiv` — C2V 生产者 FIFO 分配

```mlir
pto.talloc_to_aiv {(id = <N>,)? split = <S>} -> <entry_type>
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
entry = alloc_c2v_producer_slot(id, split)
// 在 Cube 侧为 C2V 方向分配一个 GlobalTensor FIFO 生产者条目
```

**参数：** 无操作数。

**返回值：** `!pto.tensor_view<...>` — FIFO 条目描述符。

**属性：**

- `id` — 管线 ID（`i32`），默认值为 `0`。当存在多条管线时用于区分。
- `split` — 分割因子（`i8`）。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须位于 `pto.section.cube` 或 Cube kernel 函数内部。
  - 必须存在对应的 `pto.aic_initialize_pipe` 且 `dir_mask` 含 C2V 位。

**示例：**

```mlir
%entry = pto.talloc_to_aiv {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
```

---

### `pto.talloc_to_aic` — V2C 生产者 FIFO 分配

```mlir
pto.talloc_to_aic {(id = <N>,)? split = <S>} -> <entry_type>
```

**产品支持情况：**

- Ascend 950PR&950DT系列产品：支持
- Atlas A3系列产品：支持
- Atlas A2系列产品：支持

**语义：**

```text
entry = alloc_v2c_producer_slot(id, split)
// 在 Vector 侧为 V2C 方向分配一个 GlobalTensor FIFO 生产者条目
```

**参数：** 无操作数。

**返回值：** `!pto.tensor_view<...>` — FIFO 条目描述符。

**属性：**

- `id` — 管线 ID（`i32`），默认值为 `0`。
- `split` — 分割因子（`i8`）。

**约束：**

- **实现检查（Atlas A2系列产品/Atlas A3系列产品/Ascend 950PR&950DT系列产品）**
  - 必须位于 `pto.section.vector` 或 Vector kernel 函数内部。
  - 必须存在对应的 `pto.aiv_initialize_pipe` 且 `dir_mask` 含 V2C 位。

**示例：**

```mlir
%entry = pto.talloc_to_aic {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf16>
```
