# docs/mkdocs 文档构建说明

本目录用于构建 PTO Tile Lib 的在线文档与本地静态文档站点，基于 MkDocs。

## 文档内容

构建后的文档是经过筛选的 PTO 指令集架构手册站点，覆盖以下内容：

- PTO 指令集架构模型章节与指令集契约
- Tile、Vector、Scalar/Control、Communication 指令参考
- 作为“语法与操作数”章节一部分的 PTO-AS 拼写
- 公开手册需要的快速开始入口页

文档的**作者维护源**主要位于主仓库文档树中：

- `docs/isa/`：PTO 指令集架构手册与分组指令参考
- `docs/isa/syntax-and-operands/`：PTO-AS 写法与操作数语法
- `docs/getting-started*.md`：公开快速开始入口页

`docs/mkdocs/src/` 只保存 MkDocs 自己维护的着陆页与站点资源。构建期虚拟页面由
`gen_pages.py` 生成，不应提交到 `docs/mkdocs/src/` 作为源码。

## 推荐方式

- 在线查看文档：访问 [文档中心](https://pto-isa.gitcode.com)
- 本地预览与离线查看：使用 MkDocs 本地构建

## 前置条件

- Python >= 3.8
- pip

建议先创建独立的 Python 虚拟环境。

## 方式一：使用 MkDocs CLI

### 1. 安装依赖

```bash
python -m pip install -r docs/mkdocs/requirements.txt
```

### 2. 本地预览

```bash
python -m mkdocs serve -f docs/mkdocs/mkdocs.yml
```

启动后可在 `http://127.0.0.1:8000` 访问文档，本地修改会自动热更新。

### 3. 构建静态站点

```bash
python -m mkdocs build -f docs/mkdocs/mkdocs.yml
```

构建输出位于仓库根级的 `site/` 目录。

## 方式二：通过 CMake 构建

适用于希望将文档构建集成到开发流程或 CI/CD 中的场景。

### 1. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv-mkdocs
source .venv-mkdocs/bin/activate  # Windows: .venv-mkdocs\Scripts\Activate.ps1
python -m pip install -r docs/mkdocs/requirements.txt
```

### 2. 配置并构建

```bash
cmake -S docs -B build/docs -DPython3_EXECUTABLE=$PWD/.venv-mkdocs/bin/python
cmake --build build/docs --target pto_docs
```

Windows（PowerShell）：

```powershell
cmake -S docs -B build/docs -DPython3_EXECUTABLE="$PWD\.venv-mkdocs\Scripts\python.exe"
cmake --build build/docs --target pto_docs
```

构建输出位于 `build/docs/site/`。

## 目录说明

- `mkdocs.yml`：MkDocs 配置文件
- `requirements.txt`：文档构建依赖
- `src/`：仅保存 MkDocs 自身维护的着陆页与站点资源
- `gen_pages.py`：文档页面生成脚本
- `check_mkdocs.py`：文档构建检查脚本

## 相关文档

- [根目录 README_zh](../../README_zh.md)
- [快速开始指南](../getting-started_zh.md)
- [文档入口](../isa/scalar/ops/micro-instruction/README_zh.md)
