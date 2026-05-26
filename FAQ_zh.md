# FAQ

## 1. 调试模式（-d）下源码修改不生效

### 问题现象

执行算子集成测试时加上 `-d` 参数后，对 `include/pto/` 下源码的修改（如添加 `cce::printf` 日志）不生效：

```bash
python3 tests/script/run_st.py -r npu -v a3 -t tadds -d
```

### 根因

`-d` 参数开启调试模式，会在 `tests/npu/a2a3/src/st/CMakeLists.txt` 中启用 `--cce-enable-print` 编译选项。该选项会改变 Bisheng 编译器的头文件搜索优先级，使编译器优先从 `$ASCEND_HOME_PATH/include/` 查找头文件，导致 CMakeLists.txt 中设置的 include 路径失效——实际编译使用的是 CANN 包内置的 pto 头文件，而非本地源码。

### 解决方案

将 CANN 包中的 pto 头文件目录重命名，强制编译器使用本地源码：

```bash
sudo mv $ASCEND_HOME_PATH/include/pto $ASCEND_HOME_PATH/include/pto_bak
```

> `$ASCEND_HOME_PATH` 通常在 `/usr/local/Ascend` 等系统目录下，需 `sudo` 权限。

调试完成后恢复：

```bash
sudo mv $ASCEND_HOME_PATH/include/pto_bak $ASCEND_HOME_PATH/include/pto
```
