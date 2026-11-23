# FAQ

## 1. Source changes not taking effect in debug mode (`-d`)

### Symptom

When running ST integration tests with the `-d` flag, local changes to `include/pto/` source files (e.g., adding `cce::printf` logs) do not take effect:

```bash
python3 tests/script/run_st.py -r npu -v a3 -t tadds -d
```

### Root Cause

The `-d` flag enables debug mode, which adds the `--cce-enable-print` compiler option in `tests/npu/a2a3/src/st/CMakeLists.txt`. This option changes the Bisheng compiler's header search priority, causing it to look in `$ASCEND_HOME_PATH/include/` first. As a result, the include paths set in CMakeLists.txt are overridden, and the compiler uses the built-in pto headers from the CANN package instead of the local source code.

### Solution

Rename the CANN package's pto header directory to force the compiler to use local sources:

```bash
sudo mv $ASCEND_HOME_PATH/include/pto $ASCEND_HOME_PATH/include/pto_bak
```

> `$ASCEND_HOME_PATH` is typically under `/usr/local/Ascend` and requires `sudo`.

Restore after debugging:

```bash
sudo mv $ASCEND_HOME_PATH/include/pto_bak $ASCEND_HOME_PATH/include/pto
```
