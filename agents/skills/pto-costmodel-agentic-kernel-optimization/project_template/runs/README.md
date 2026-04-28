# 运行产物目录

```text
runs/
  _template/
    files.txt
    notes.md
  iter-001/
    files.txt
    patch.diff
    notes.md
```

每轮**必须**保存三个文件，不允许缺失：

- `files.txt`：本轮修改的文件列表（调参类迭代写"环境变量调参，无源文件修改"）
- `patch.diff`：本轮代码 diff（调参类迭代写参数变更说明，见 `task.md` 中的格式要求）
- `notes.md`：本轮详细记录（Pre-Edit / Changes / Post-Run / Analysis / Next）

从 `runs/_template/` 复制后填写。
