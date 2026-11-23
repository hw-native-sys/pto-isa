# pkg_inc/

该目录用于存放**非对外暴露（内部使用）的头文件**，遵循 CANN 打包约定：

- **对外**、构成公共 API 的头文件放在 `include/` 下；
- **非对外**、仅内部模块使用的头文件放在 `pkg_inc/` 下。

## 打包与安装

`cmake/package.cmake` 会将该目录与 `include/` 并列安装到 `<arch>-linux/pkg_inc`。
CANN 安装器已自动创建顶层 `pkg_inc -> <arch>-linux/pkg_inc` 软链接，因此新增的
内部头文件安装后即可在 `<安装路径>/pkg_inc/pto/...` 访问。

新增内部头文件应放到对应的 `pto/` 子目录下，而不是 `include/pto/`；已经放在
`include/` 下的头文件保持不变。

## 目录结构

新增内部头文件与 `include/pto/` 的布局保持一致：

- `pto/comm/`     ：通信内部实现
- `pto/common/`   ：平台无关的共享内部实现
- `pto/costmodel/`：cost model 内部实现
- `pto/cpu/`      ：CPU 侧内部实现
- `pto/npu/`      ：NPU 侧内部实现（按 SoC 代际划分）
