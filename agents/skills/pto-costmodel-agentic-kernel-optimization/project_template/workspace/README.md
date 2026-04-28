# 工作区说明

`workspace/` 用于存放当前算子项目实际修改的 PTO 代码副本。

原则：

- 只在这里修改代码
- 不直接修改原始 `pto-kernels` 目录
- 一个项目只服务于一个算子

## 初始化方式

将原始代码复制到：

```bash
workspace/pto-kernels
```

示例：

```bash
cp -R <original_pto_kernels_path> ./workspace/pto-kernels
```
