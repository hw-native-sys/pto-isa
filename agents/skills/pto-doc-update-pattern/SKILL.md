---
name: PTO指令文档更新模式
description: PTO ISA 新增指令时需要更新的文档文件和位置模式总结。根据 docs/assembly/README.md 中的分类动态选择需要修改的汇编文件。触发：新增 PTO 指令（如 TPOW、TPOWS）后需要同步更新文档时。
license: CANN Open Software License Agreement Version 2.0
---

# PTO指令文档更新模式

此 skill 总结了为 PTO ISA 添加新指令（如 TPOW、TPOWS）时需要更新的文档文件和位置模式。

## 适用场景

当新增 PTO 指令后，需要同步更新相关文档时使用此 skill。使用 `git status` 查看当前 staging 的文件列表来确认所有需要修改的文件。

## Git Staged 文件分类

### 新增文件（需预先创建）

| 类型 | 说明 |
|------|------|
| 指令diagrams | `docs/figures/isa/{指令名}.svg` - 指令操作示意图 |
| 指令文档 | `docs/isa/{指令名}.md` - 详细指令文档（英文） |
| 指令文档 | `docs/isa/{指令名}_zh.md` - 详细指令文档（中文） |

### 修改文件（需同步更新）

| 类型 | 说明 |
|------|------|
| ISA主索引 | `docs/PTOISA.md` - ISA索引表格 |
| ISA主索引 | `docs/PTOISA_zh.md` - ISA索引表格（中文） |
| ISA参考目录 | `docs/isa/README.md` - 按分类排序的指令列表 |
| ISA参考目录 | `docs/isa/README_zh.md` - 按分类排序的指令列表（中文） |
| 菜单文档 | `docs/menu_apis.md` - 按分类排序的中文链接 |
| 汇编参考 | `docs/assembly/<类别>-ops.md` - 根据指令类型动态选择 |
| 指令族矩阵 | `docs/mkdocs/src/manual/appendix-d-instruction-family-matrix.md` - 指令族矩阵 |
| 指令族矩阵 | `docs/mkdocs/src/manual/appendix-d-instruction-family-matrix_zh.md` - 指令族矩阵（中文） |
| include索引 | `include/README.md` - 实现状态表格 |
| include索引 | `include/README_zh.md` - 实现状态表格（中文） |

---

## 动态选择汇编文档

根据 `docs/assembly/README.md` 中的 **### 2. PTO Tile Operation Categories** 决定需要修改的汇编文件。

### 汇编文件列表（对应分类）

| 分类 | 汇编文件 | 说明 |
|------|----------|------|
| Elementwise (Tile-Tile) | `elementwise-ops.md` | tile-tile 逐元素操作 |
| Tile-Scalar / Tile-Immediate | `tile-scalar-ops.md` | tile-标量操作 |
| Axis Reduce / Expand | `axis-ops.md` | 轴归约/扩展操作 |
| Memory (GM ↔ Tile) | `memory-ops.md` | 内存操作 |
| Matrix Multiply | `matrix-ops.md` | 矩阵乘操作 |
| Data Movement / Layout | `data-movement-ops.md` | 数据搬运/布局操作 |
| Complex | `complex-ops.md` | 复杂操作 |
| Manual Resource Binding | `manual-binding-ops.md` | 手动资源绑定操作 |
| Scalar Arithmetic | `scalar-arith-ops.md` | 标量算术操作 |
| Control Flow | `control-flow-ops.md` | 控制流操作 |
| Auxiliary Functions | `nonisa-ops.md` | 辅助函数操作 |

### 动态选择规则

1. **确定指令分类** - 查看 `docs/assembly/README.md` 中的分类定义
2. **选择对应文件** - 根据分类选择对应的 `<类别>-ops.md` 文件
3. **更新计数** - 更新该文件中 `**Total Operations:** N` 的计数
4. **添加章节** - 在该分类最后一个指令后插入新指令章节

---

## 更新模式详解

### 1. ISA 主索引文件

#### docs/PTOISA.md / docs/PTOISA_zh.md
- **位置**: 指令索引表格
- **分类**: 
  - 逐元素（Tile-Tile）指令 → 插在 `TFMOD` 后
  - Tile-标量 / Tile-立即数 → 插在 `TSUBSC` 后

#### include/README.md / include/README_zh.md
- **位置**: 实现状态表格（按字母序）
- **分类**:
  - TPOW → 插在 `TPRELU` 和 `TPUT` 之间
  - TPOWS → 插在 `TPUT_ASYNC` 和 `TQUANT` 之间

### 2. ISA 参考目录

#### docs/isa/README.md / docs/isa/README_zh.md
- **位置**: 按分类排序的指令列表
- **分类**:
  - Elementwise (Tile-Tile) → 插在 `TFMOD` 后
  - Tile-Scalar / Tile-Immediate → 插在 `TSUBSC` 后

### 3. 菜单文档

#### docs/menu_apis.md
- **位置**: 按分类排序的中文链接列表
- **同 ISA 参考目录结构**

### 4. 汇编文档（动态选择）

根据指令类型选择对应的文件：

| 指令类型 | 目标文件 | 插入位置 |
|----------|----------|----------|
| TPOW (Elementwise) | `elementwise-ops.md` | TFMOD 后 |
| TPOWS (Tile-Scalar) | `tile-scalar-ops.md` | TSU BSC 后 |
| TROWSUM (Axis) | `axis-ops.md` | 最后一个 Axis 指令后 |
| TLOAD (Memory) | `memory-ops.md` | 最后一个 Memory 指令后 |
| TMATMUL (Matrix) | `matrix-ops.md` | 最后一个 Matrix 指令后 |
| TMOV (Data Movement) | `data-movement-ops.md` | 最后一个 Data Movement 指令后 |
| TQUANT (Complex) | `complex-ops.md` | 最后一个 Complex 指令后 |
| TASSIGN (Manual Binding) | `manual-binding-ops.md` | 最后一个 Manual 指令后 |

### 5. 指令族矩阵

#### docs/mkdocs/src/manual/appendix-d-instruction-family-matrix.md
- **位置**: D.2 覆盖统计表 + D.4 家族矩阵表
- **D.2 更新示例**:
  ```
  | Elementwise (Tile-Tile) | 28 → 29 |
  | Tile-Scalar / Tile-Immediate | 19 → 20 |
  | Total | 126 → 128 |
  ```
- **D.4 更新**:
  - 在对应分类的最后一条目后插入新指令

#### docs/mkdocs/src/manual/appendix-d-instruction-family-matrix_zh.md
- **同英文版本**

---

## 常见新增指令分类与插入位置

### Tile-Tile (逐元素双Tile)
- **插入位置**: `TFMOD` 之后
- **对应文件**: `elementwise-ops.md`
- **示例**: TPOW

### Tile-Scalar (Tile与标量)
- **插入位置**: `TSUBSC` 之后
- **对应文件**: `tile-scalar-ops.md`
- **示例**: TPOWS

### Axis Reduce / Expand
- **插入位置**: 最后一个 Axis 指令之后
- **对应文件**: `axis-ops.md`

### Memory (GM ↔ Tile)
- **插入位置**: 最后一个 Memory 指令之后
- **对应文件**: `memory-ops.md`

---

## 更新检查清单

### 新增文件（预先创建）
- [ ] `docs/figures/isa/{新指令}.svg` - 指令操作示意图
- [ ] `docs/isa/{新指令}.md` - 详细指令文档（英文）
- [ ] `docs/isa/{新指令}_zh.md` - 详细指令文档（中文）

### 修改文件（同步更新）
- [ ] `docs/PTOISA.md` - ISA主索引
- [ ] `docs/PTOISA_zh.md` - ISA主索引（中文）
- [ ] `include/README.md` - include索引
- [ ] `include/README_zh.md` - include索引（中文）
- [ ] `docs/isa/README.md` - ISA参考目录
- [ ] `docs/isa/README_zh.md` - ISA参考目录（中文）
- [ ] `docs/menu_apis.md` - 菜单文档
- [ ] `docs/assembly/<类别>-ops.md` - 动态选择的汇编文件（英文+中文）
- [ ] `docs/mkdocs/src/manual/appendix-d-instruction-family-matrix.md` - 指令族矩阵
- [ ] `docs/mkdocs/src/manual/appendix-d-instruction-family-matrix_zh.md` - 指令族矩阵（中文）

---

## 注意事项

1. **英文+中文**: 每个文件都有中英文两个版本，需要同步更新
2. **动态选择**: 根据 `docs/assembly/README.md` 选择的分类来确定需要修改的汇编文件
3. **计数变化**: 需要同时更新 Operation Count（分类小计）和 Total（总计）
4. **详细指令文档**: 需要预先创建在 `docs/isa/` 目录下
5. **diagrams**: 需要预先创建在 `docs/figures/isa/` 目录下
6. 使用 `git status` 可以查看当前 staging 的文件列表，这是确认所有需要修改文件的最佳方式