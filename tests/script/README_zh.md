# tests/script/

用于构建与运行仓库测试套件的入口脚本。

## NPU ST（sim / npu）

- 构建 + 运行：`tests/script/run_st.py`
- 仅构建：`tests/script/build_st.py`

常用参数：

- `-r, --run-mode`：`sim` 或 `npu`
- `-v, --soc-version`：`a3` 或 `a5`（映射到内部的 `SOC_VERSION`）
- `-t, --testcase`：testcase 名（例如 `tmatmul`）
- `-g, --gtest_filter`：可选 gtest 过滤器（运行单个 case）
- `-d, --debug-enable`：可选 Debug 构建（仅 `run_st.py` 支持）

示例：

```bash
python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
```

## CPU ST

- 批量构建 + 运行：`tests/script/all_cpu_tests.py`

选项：

- `-v, --verbose`：打印构建/运行输出
- `-b, --build-folder`：构建目录（默认：`build_tests`）

示例：

```bash
python3 tests/script/all_cpu_tests.py --verbose
```

## 便捷封装脚本

- 推荐测试集：`tests/run_st.sh`
- CPU 测试：`tests/run_cpu_tests.sh`

最新参数请以 `python3 <script> -h` 输出为准。
