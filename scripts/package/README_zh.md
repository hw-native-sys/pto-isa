# scripts/package/

PTO Tile Lib 的打包/发布脚本与模板，用于生成可分发产物。

## 目录结构（概览）

- `pto_isa/`：PTO Tile Lib 专用的打包模板与辅助脚本
- `module/`：模块级打包描述
- `common/`：共享辅助工具与配置片段

## 构建与安装

### 打包

```bash
cd ${git_clone_path}
./build.sh --pkg
```

构建完成后，`.run` 安装包生成在 `scripts/package/output/` 目录下。

### 安装

通过 `.run` 自解压包安装，需指定安装类型（三选一）：

| 参数 | 说明 |
|------|------|
| `--full` | 完整安装，包含头文件、库、测试资源等全部内容 |
| `--run` | 运行时安装，仅安装运行所需的库和依赖 |
| `--devel` | 开发环境安装，包含头文件和库 |

```bash
# 完整安装到指定路径
./scripts/package/output/pto_isa_*.run --full --install-path=/your/install/path

# 静默安装，跳过交互确认（适用于 CI/CD 等非交互环境）
./scripts/package/output/pto_isa_*.run --full --quiet --install-path=/your/install/path

# 仅安装运行时组件
./scripts/package/output/pto_isa_*.run --run --install-path=/your/install/path
```

常用安装参数：

| 参数 | 说明 |
|------|------|
| `--install-path=<path>` | 指定安装目标目录 |
| `--quiet` | 静默模式，跳过交互确认（适用于非交互环境） |
| `--install-for-all` | 为所有用户安装 |
| `--uninstall` | 卸载已安装的产品 |
| `--upgrade` | 升级已有安装 |
| `--version` | 查看包版本信息 |
| `--pre-check` | 安装前预依赖检查 |

更多参数可通过 `--help` 查看：

```bash
./scripts/package/output/pto_isa_*.run --help
```
