# CostModel 后端说明（打桩 stub / 拟合 fit）

本文档说明仓库内 CostModel 的两类后端：打桩（stub）与拟合（fit）。

## 总览

CostModel 路径通过 `__COSTMODEL` 使能，当前后端分为：

- `stub`：基础 CostModel 指令行为与覆盖验证路径
- `fit`：基于公式拟合的时延预测路径

当前测试套件命名与后端术语映射：

- `st` => `stub`
- `st_fit` => `fit`

对应目录：

- `tests/costmodel/st/`
- `tests/costmodel/st_fit/`

## 代码映射

- CostModel 通用入口与类型：
  - `include/pto/costmodel/pto_instr.hpp`（打桩使用）
  - `include/pto/costmodel/lightweight_costmodel.hpp`（拟合使用）
- fit 公式后端实现：
  - `include/pto/costmodel/a2a3/formula_costmodel/formula_backend_compute.hpp`
  - `include/pto/costmodel/a2a3/formula_costmodel/formula_backend_transfer.hpp`

## 打桩后端（stub，基础路径）

- CMake 工程：`tests/costmodel/st/CMakeLists.txt`
- 重点：
  - 指令级 CostModel 结果验证
  - 支持算子的基础行为检查

## 拟合后端（fit）

- CMake 工程：`tests/costmodel/st_fit/CMakeLists.txt`
- 重点：
  - 基于参数公式的 cycles/latency 估算
  - 运行时配置对结果影响的验证（频率、带宽、搬运路径）

## 公式参数生成

执行拟合后端（`st_fit` 套件）时，运行脚本会从 CSV 生成公式参数头文件：

- 生成脚本：`include/pto/costmodel/a2a3/formula_costmodel/gen_formula_params_header.py`
- 输入文件：`include/pto/costmodel/a2a3/formula_costmodel/formula_params.csv`
- 输出文件：`include/pto/costmodel/a2a3/formula_costmodel/formula_params_generated.hpp`

打桩后端（`st`）不需要该生成步骤。

## 运行命令

单套件/单用例：

```bash
python3 tests/run_costmodel.py --suite st --testcase tadd --clean --verbose
python3 tests/run_costmodel.py --suite st_fit --testcase time_predict --clean --verbose
```

批量执行（自动覆盖 `st` + `st_fit`）：

```bash
bash tests/run_costmodel_tests.sh
```
