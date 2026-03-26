# scripts/

仓库辅助脚本，主要用于打包/发布相关流程。

## 目录结构

- `package/`：打包脚本（Python + 模板 + 配置）
- `gen_pto_isa_capability_manifest.py`：导出 PTO ISA 指令、公开 intrinsic 以及前端数据类型暴露情况的 JSON 能力清单

## 入口

- `build.sh --pkg` 会触发 `scripts/package/` 下实现的打包流程
- `python3 scripts/gen_pto_isa_capability_manifest.py --output /tmp/pto-isa-capability.json` 可为下游工具链 gate 生成能力清单
