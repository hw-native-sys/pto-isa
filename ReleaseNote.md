# 发布说明 — PTO Tile Lib

本文档用于汇总 PTO Tile Lib 的版本变更信息。

格式遵循 Keep a Changelog 风格（Added / Changed / Fixed / Deprecated / Removed / Security）。

## Unreleased

- 暂无

### Added

- PTO Tile Lib 初次公开发布。

### Changed

- 暂无

### Fixed

- 暂无

### Deprecated

- 暂无

### Removed

- 暂无

### Security

- 漏洞反馈流程请参见 `SECURITY.md`。

## 兼容性说明

- **Ascend（NPU / simulator）**：依赖 Ascend CANN toolkit `>= 8.3`（详见 `version.info`）；具体支持的 SoC 与工具链版本取决于已安装的 CANN 发行版。
- **CPU simulator**：支持在 macOS / Linux / Windows 上结合 C++ 工具链与 Python 运行；详见 `docs/zh/getting-started.md`。
