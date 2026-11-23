# 自定义 PyTorch 算子（KERNEL_LAUNCH）示例

本示例展示如何实现一个基于 PTO 的自定义 kernel，并通过 `torch_npu` 将其暴露为 PyTorch 算子。

## 目录结构

```
demos/baseline/add/
├── op_extension/              # Python 包入口（模块加载）
├── csrc/
│   ├── kernel/                # PTO kernel 实现
│   └── host/                  # Host 侧 PyTorch 算子注册
├── test/                      # 最小化 Python 测试
├── CMakeLists.txt             # 构建配置
├── setup.py                   # Wheel 构建脚本
└── README.md                  # 本文档
```

## 1. 实现 kernel

在 `demos/baseline/add/csrc/kernel/` 下新增 kernel 源码，并将其加入构建。例如要构建 `add_custom.cpp`，需要在 `demos/baseline/add/CMakeLists.txt` 中添加：

```cmake
ascendc_library(no_workspace_kernel STATIC
    csrc/kernel/add_custom.cpp
)
```

构建选项与细节请参考昇腾社区文档：https://www.hiascend.com/ascend-c

## 2. 与 PyTorch 集成（`torch_npu`）

Host 侧实现位于 `demos/baseline/add/csrc/host/`。

### 2.1 定义算子 schema（Aten IR）

PyTorch 使用 `TORCH_LIBRARY` / `TORCH_LIBRARY_FRAGMENT` 声明算子 schema，使其可从 Python 通过 `torch.ops.<namespace>.<op_name>` 调用。

示例：在 `npu` 命名空间注册一个自定义 `my_add` 算子：

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("my_add(Tensor x, Tensor y) -> Tensor");
}
```

之后 Python 可通过 `torch.ops.npu.my_add` 调用。

### 2.2 实现算子

1. 引入由构建系统生成的 kernel launch 头文件 `aclrtlaunch_<kernel_name>.h`。
2. 按需分配输出张量/工作区（workspace）。
3. 通过 `ACLRT_LAUNCH_KERNEL`（在本示例中由 `EXEC_KERNEL_CMD` 封装）将 kernel 入队执行。

```cpp
#include "utils.h"
#include "aclrtlaunch_add_custom.h"

at::Tensor run_add_custom(const at::Tensor &x, const at::Tensor &y)
{
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 20;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    EXEC_KERNEL_CMD(add_custom, blockDim, x, y, z, totalLength);
    return z;
}
```

### 2.3 注册实现

使用 `TORCH_LIBRARY_IMPL` 注册实现。对 NPU 执行而言，`torch_npu` 使用 `PrivateUse1` dispatch key，关于 `PrivateUse1` 的详细介绍请参考 PyTorch 官方文档：
https://docs.pytorch.org/tutorials/advanced/privateuseone.html

```cpp
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("my_add", TORCH_FN(run_add_custom));
}
```

## 3. 构建与运行

本示例依赖 PTO Tile Lib、PyTorch、`torch_npu` 与 CANN。请参考 `torch_npu` 官方安装指南：

https://gitcode.com/ascend/pytorch#%E5%AE%89%E8%A3%85

或执行：

```bash
python3 -m pip install -r requirements.txt
```

### 3.1 设置目标 SoC

编辑 `demos/baseline/add/CMakeLists.txt`，把 `SOC_VERSION` 设置为目标芯片（例如 A2/A3 使用 `Ascend910B1`）：

```cmake
set(SOC_VERSION "Ascendxxxyy" CACHE STRING "system on chip type")
```

可在目标机器上执行 `npu_smi info` 查询芯片名称，并按 `Ascend<Chip Name>` 的形式填写。

### 3.2 构建 wheel

设置 PTO Tile Lib 路径并构建 wheel：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa
rm -rf build op_extension.egg-info
python3 setup.py bdist_wheel
```

### 3.3 安装 wheel

```bash
cd dist
pip uninstall *.whl
pip install *.whl
```

### 3.4 运行测试

```bash
cd test
python3 test.py
```
