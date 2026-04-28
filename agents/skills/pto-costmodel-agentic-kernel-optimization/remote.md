---
name: ako4pto-remote-bringup
description: 用于 AKO4PTO 的远程 Ascend 环境 bring-up。覆盖 SSH 连通、工作区上传、Python 选择、依赖安装、环境自检及 benchmark 验证。
---

# AKO4PTO 远程环境 Bring-Up

## 概述

把远程 Ascend 机器带到"可以稳定运行目标 PTO benchmark"的状态。覆盖两种情况：已有环境的快速复用、空环境的从零 bring-up。

本文档只负责远程环境准备和排障，调优流程和迭代协议以 `task.md` 为准。

> [!NOTE]
> `<user>`、`<remote_ip>`、`<path_to>` 由用户在每次会话开始时提供，不要硬编码。

## 工作流

### 1. 确认 SSH

```bash
ssh -o BatchMode=yes <user>@<remote_ip> "echo ssh_key_ok"
```

### 2. 本地准备工作区

向用户确认 `pto-kernels` 本地路径后，**先在本地**确保 `external/src/` 下的四个依赖库齐全：

```bash
cd <pto-kernels路径>
ls external/src/pto-dsl external/src/PTOAS external/src/pto-isa external/src/ops-transformer
```

缺失则在本地补全：

```bash
bash scripts/bootstrap_workspace.sh
```

这样后续上传时 `external/src/` 会一并传到远程，避免远程 git clone 受网络影响。

### 3. 上传工作区

远程目录结构与本地保持一致。上传命令（优先 scp，备选 rsync）：

```bash
# 首选
scp -r projects/<operator_name>/workspace/pto-kernels <user>@<remote_ip>:<path_to>/AKO4PTO/projects/<operator_name>/workspace/

# 备选（增量同步）
rsync -avz projects/<operator_name>/workspace/pto-kernels/ <user>@<remote_ip>:<path_to>/AKO4PTO/projects/<operator_name>/workspace/pto-kernels/
```

### 4. 识别远程 Python

```bash
which python3 && python3 --version
```

- 不要假设系统 `python3` 满足 PTO 依赖，不要替换系统 Python
- 如果远程默认是 `Python 3.9`，优先使用用户态 `Python 3.11`（如 `~/.local/miniforge3/envs/pto-py311/bin/python`）
- 一旦选定解释器，后续所有操作都固定使用，不和系统 Python 混用

### 5. 快速复用（已有环境）

如果远程环境大概率已搭好，先做最小校验，不要急着安装任何东西：

```bash
cd <path_to>/AKO4PTO/projects/<operator_name>/workspace/pto-kernels
bash scripts/source_env.sh
<python_exec> scripts/check_env.py --strict
<python_exec> -c "import mlir.ir; from mlir.dialects import pto; print('mlir_ok')"
<python_exec> -c "import torch, torch_npu; print(torch.npu.is_available())"
ptoas --version
<python_exec> -m pto_kernels.bench.runner <target_spec>
```

全部通过 → 复用成功，直接进入调优。任一步失败 → 进入下面的完整 bring-up（步骤 6 起）。

### 6. 确认远程依赖源码

步骤 2 已在本地确保 `external/src/` 齐全。上传后在远程确认：

```bash
ls external/src/pto-dsl external/src/PTOAS external/src/pto-isa external/src/ops-transformer
```

缺失则在远程执行 `bash scripts/bootstrap_workspace.sh`。

> [!NOTE]
> `bootstrap_workspace.sh` 需要远程能访问 GitHub / GitCode。超过 10 秒无响应 → 在本地补全后重新 scp 上传 `external/src/`。

### 7. 安装 Python 依赖

```bash
<python_pip> install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
<python_pip> install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-npu==2.8.0.post2 --extra-index-url https://download.pytorch.org/whl/cpu
<python_pip> install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy PyYAML mpmath 'pybind11<3' decorator
```

> [!IMPORTANT]
> **10 秒超时规则**：远程 `pip install` 或任何下载命令超过 **10 秒无响应**，立即 Ctrl-C，切换为本地下载 + scp 上传：
> ```bash
> pip download -i https://pypi.tuna.tsinghua.edu.cn/simple <package> -d /tmp/wheels/
> scp -r /tmp/wheels/ <user>@<remote_ip>:<path_to>/wheels/
> <python_pip> install --no-index --find-links <path_to>/wheels/ <package>
> ```

### 8. 加载环境 + 检查 ptoas/mlir

```bash
bash scripts/source_env.sh
ptoas --version
<python_exec> -c "import mlir.ir; from mlir.dialects import pto; print('mlir_ok')"
```

如果失败，通常需要手动补环境变量：

```bash
export PATH=<py311_bin>:<ptoas_cli_bin>:$PATH
export LD_LIBRARY_PATH=<py311_lib>:<mlir__mlir_libs>:<ptoas_cli_lib>:/usr/local/Ascend/cann-*/aarch64-linux/lib64:$LD_LIBRARY_PATH
```

注意：`mlir/_mlir_libs` 和 `ptoas` CLI 的 `lib` 目录往往都必须放进 `LD_LIBRARY_PATH`。

### 9. 环境自检 + benchmark 验证

执行与步骤 5 相同的完整校验序列（`check_env.py --strict` → mlir → torch_npu → ptoas → bisheng），然后跑一次目标 benchmark：

```bash
<python_exec> -m pto_kernels.bench.runner <target_spec>
```

环境是否可用，以 `task.md` 里的 gate 为准。

## 排障

- **SSH 要密码**：检查本地私钥、远程 `authorized_keys`、登录用户名。

- **已有环境跑不通**：先确认仓库路径、Python 路径、`source_env.sh` 是否成功、`check_env.py --strict`。只有确认不可修复时才转入完整 bring-up。

- **`torch_npu` 导入失败**：检查 Ascend 环境是否已加载、Python 路径是否正确、`torch_npu` 是否装到了正确的解释器。

- **`torch.npu.is_available()` 为 False**：检查 `LD_LIBRARY_PATH` 是否包含 CANN `aarch64-linux/lib64`，`source_env.sh` 是否提前退出。

- **`import mlir.ir` 失败**：检查 `ptoas` wheel 是否装到了当前解释器，`LD_LIBRARY_PATH` 是否包含 `mlir/_mlir_libs`。

- **`ptoas --version` 通过但编译失败**：检查 `PATH` 和 `LD_LIBRARY_PATH` 是否包含 ptoas CLI 的 bin/lib。

- **远程下载超过 10 秒无响应**：不要重试。本地 `pip download` → scp 上传 → 远程 `pip install --no-index --find-links` 离线安装。
