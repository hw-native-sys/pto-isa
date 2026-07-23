# PTO-ISA对外头文件和库文件说明

#### 接口分类

PTO 全称 Parallel Tile Operation，即并行 Tile 操作，是 Ascend CANN 定义的面向 Tile 编程的虚拟指令集架构。本仓在该架构下共提供 124 条 Tile 指令，其中包含一套片间通信扩展指令。这些指令覆盖逐元素计算、归约、广播、矩阵乘与 GEMV、数据搬运、类型转换、布局变换、排序、Union 计算等计算与变换场景，以及同步、资源配置等系统控制场景，可供上层框架、算子实现与编译工具链统一调用。

PTO 指令统一采用 Tile 块级抽象，命名风格为：指令类别前缀+计算名称，整体首字母大写，采用 PascalCase 风格。其中绝大多数指令以 `T` 为前缀，表示面向 Tile 对象的操作；矩阵级不规则访存指令以 `M` 为前缀，如 `MGATHER`、`MSCATTER`；跨核同步屏障为 `SYNCALL`。下文为了描述方便，将本文中的接口统称为 PTO 接口。

**表 1 关键指令类别**

| 指令类别 | 描述 |
| --- | --- |
| 计算与搬运 | 以 `T` 为前缀的 Tile 级核心指令，涵盖 Tile 与 Tile、Tile 与标量的元素计算、类型转换、选择、行/列/部分归约、矩阵乘与 GEMV、数据搬运 Load/Store/Mov/Gather/Scatter/Extract/Insert 等，如 `TADD`、`TMUL`、`TMATMUL`、`TGEMV`、`TLOAD`、`TSTORE`、`TMOV`、`TGATHER`、`TSCATTER`、`TCVT`。 |
| 片间通信 | 同样以 `T` 为前缀的 NPU 间通信与同步指令，覆盖点对点通信、信号同步与集合通信，支持同步与异步两种形式，如 `TGET`、`TGET_ASYNC`、`TPUT`、`TPUT_ASYNC`、`TNOTIFY`、`TWAIT`、`TTEST`、`TBROADCAST`、`TREDUCE`。 |
| 矩阵级访存 | 以 `M` 为前缀的矩阵级 Gather/Scatter 指令，如 `MGATHER`、`MSCATTER`。 |
| 资源绑定 | `TASSIGN`，将 Tile 对象手动绑定到实现定义的片上地址，对应 Manual 手动放置模式。 |
| 同步屏障 | `SYNCALL`，跨核同步屏障指令，用于多核间的执行同步。 |

---

#### 调用接口依赖的头文件和库文件说明

PTO 接口的头文件位于 CANN 安装目录下的 `<arch>-linux/include/` 子目录中，例如 `${INSTALL_DIR}/aarch64-linux/include/pto/pto-inst.hpp`。`${INSTALL_DIR}` 请替换为 CANN 软件安装后的存储路径，以 root 用户安装为例，默认路径为 `/usr/local/Ascend/cann`。各头文件的用途如下表所示。

**表 2 头文件列表**

| 定义接口的头文件 | 用途 | 对应的库文件 |
| --- | --- | --- |
| `pto/pto-inst.hpp` | PTO 指令集架构的唯一对外统一入口头文件，上层只需包含 `pto/pto-inst.hpp` 即可获得全部公开接口，无需逐个包含子头文件。该头文件按职责聚合以下模块：<br>**Tile 类型系统与公共工具**：用于定义 Tile 对象、内存与缓冲区管理、常量、Kernel 元信息及通用工具。<br>**指令 API 声明**：用于定义 124 条 Tile 指令的统一声明，每条指令提供 Auto 自动放置与 Manual 手动放置两种形式，覆盖逐元素计算、归约、广播、矩阵乘与 GEMV、数据搬运、类型转换、布局变换、排序、Union 计算、同步、资源配置与片间通信等接口。<br>**多后端自动选择**：依据编译期宏自动选择实现后端，定义 `__CPU_SIM` 时启用 CPU 仿真后端，定义 `__COSTMODEL` 时启用 A2/A3 CostModel 后端，定义 `__CCE_AICORE__` 时启用 NPU 真机后端，并按 SoC 代际 A2/A3、A5、Kirin 等划分实现。<br>**片间通信指令库**：用于定义 NPU 间点对点通信、信号同步与集合通信接口，含同步与异步两套接口。 | 无。PTO-ISA 为 header-only 纯头文件模板库，指令实现以模板与内联形式在编译期展开进上层 Kernel，无需单独链接 `.so` 文件。编译期依赖 BiSheng 编译器 `bisheng-compiler` 将含 PTO 指令的 Kernel 编译为设备代码；运行期依赖 CANN Runtime 库 `libruntime.so` 提供设备执行环境。 |
