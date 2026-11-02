# PTO-ISA Host 日志

[English](README.md)

## 第一部分：需求场景设计

### 1. 范围与职责

统一 PTO-ISA Host 侧的日志接口、级别、输出格式和文件管理。调用方提供级别、模块名和正文，日志组件负责过滤、格式化及输出。

采用同步日志，面向 Linux Host，包括 CCU 的 host 控制代码。Device 日志、`PTO_ASSERT`、`static_assert` 保持原样。

### 2. 日志行为

日志级别由调用方选择，建议用途如下：

| 场景 | 级别与规则 |
| --- | --- |
| 参数校验或接口调用失败 | ERROR，记录原因、返回码和必要上下文 |
| 异常后采用回退配置继续 | WARN，记录异常和回退行为 |
| 初始化、建链、释放成功及阶段跟踪（Trace） | INFO，记录进度与阶段耗时 |
| 额外调试明细 | DEBUG，按需开启 |

模块名是调用方提供的字符串，无预定义列表或注册要求，仅作为标签。PTO 文件和 stdout 的每条日志包含时间、级别、线程 / 进程 ID、模块、源码位置和正文；rank、device、peer 等业务上下文由调用点填写。

支持文件、stdout、外部 callback 三种出口：注册 callback 后仅调用 callback；否则在文件和 stdout 中二选一。callback 接收提交给 Logger 的全部 DEBUG / INFO / WARN / ERROR 日志，绕过组件阈值，由接收方过滤。

### 3. 配置与路径

| 配置 | 规则 |
| --- | --- |
| `PTO_ISA_GLOBAL_LOG_LEVEL` | 单字符 `0 / 1 / 2 / 3 / 4` 对应 DEBUG / INFO / WARN / ERROR / 关闭文件及 stdout；默认 ERROR |
| `PTO_ISA_SLOG_PRINT_TO_STDOUT` | 仅字符串 `1` 启用 stdout，其余值使用文件；callback 优先 |

上述两个 PTO 前缀变量独立控制 PTO 日志，日志路径继续复用下表的 ASCEND 路径变量，不新增 PTO 专属路径或保留数量变量。

文件目录按以下优先级选择；候选变量未设置、为空或路径非法时尝试下一项：

| 优先级 | 环境变量及含义 | 最终目录 |
| --- | --- | --- |
| 1 | `ASCEND_PROCESS_LOG_PATH`：可选，指定日志根目录 | `${ASCEND_PROCESS_LOG_PATH}/debug/plog` |
| 2 | `ASCEND_WORK_PATH`：可选，指定工作根目录 | `${ASCEND_WORK_PATH}/log/debug/plog` |
| 3 | `HOME`：用户主目录，通常由系统设置 | `${HOME}/ascend/log/debug/plog` |
| 4 | 前三项均不可用时，使用当前工作目录 | `./ascend/log/debug/plog` |

不必同时设置这些变量；同时设置时，取优先级最高的有效值。变量值填写根目录，PTO 自动追加表中的后缀。路径配置在 Logger 首次初始化时读取并固定；相对路径按当时的工作目录解析，后续修改环境变量或切换工作目录不会改变日志目录。

例如，在启动程序前设置：

```bash
export ASCEND_PROCESS_LOG_PATH=/data/pto_logs
```

日志将写入 `/data/pto_logs/debug/plog/`，当前用户需要有创建目录和写入文件的权限。未设置两个 ASCEND 路径变量时，通常使用 `${HOME}/ascend/log/debug/plog`；`HOME` 也不可用时回退到当前工作目录。

PTO 在 plog 下使用独立文件，仍自行写入、滚动和老化。

### 4. 文件、并发与退出约束

| 项目 | 要求 |
| --- | --- |
| 文件名 | `pto_isa_<pid>_<YYYYMMDDHHMMSS>_<seq>.log`，独占创建，不覆盖已有文件 |
| 创建时机 | 第一条实际写文件的日志到来时 |
| 大小与数量 | 单文件 20 MiB；目录内全部匹配 PTO 文件目标为 64 个，包含即将创建的文件，无数量环境变量 |
| 老化 | 新建文件时按文件名时间戳清理最旧文件；不按天数或空闲时间触发，其他前缀文件不受影响 |
| 磁盘条件 | 每次新建文件前，当前用户可用空间必须大于 10 GiB |

## 第二部分：实现接口、机制及流程设计

### 1. 文件与类职责

文件位于 `include/pto/comm/logger/`，由现有 include 打包流程交付。

| 文件 | 职责 |
| --- | --- |
| `host_log.hpp` | 公共设置接口、四种日志宏、模块及源码位置拼接 |
| `host_logger.hpp` | `PtoHostLogger`：级别、callback、格式化和出口选择 |
| `host_log_file_sink.hpp` | `PtoHostLogFileSink`：目录、fd、写入、滚动、老化和文件 mutex |

依赖为 `host_log.hpp` → `host_logger.hpp` → `host_log_file_sink.hpp`。Logger 通过 `unique_ptr` 独占 FileSink；FileSink 只接收完整字符串，不感知级别、模块和 callback。

日志头可独立包含，在 `__CCE_KT_TEST__` 下拒绝编译，不进入 Device 总入口；调用方在类及 namespace 外包含 `host_log.hpp`。

### 2. 调用接口与级别

接口声明：

```cpp
namespace pto::comm::hostlog {
enum LogLevel : int32_t {
    DEBUG_LEVEL = 0, INFO_LEVEL, WARN_LEVEL, ERROR_LEVEL, FATAL_LEVEL, BUTT_LEVEL
};
enum LogStatus : int32_t {
    LOG_SUCCESS = 0, LOG_INVALID_ARGUMENT = -1, LOG_ALREADY_REGISTERED = -2
};
using ExternalLog = void (*)(int32_t level, const char* message);
inline int32_t PtoSetHostLogLevel(int32_t level);
inline int32_t PtoSetExternalLogger(ExternalLog func);
}
```

调用示例：

```cpp
#include "pto/comm/logger/host_log.hpp"

PTO_HOST_LOG_DEBUG("MY_MODULE", "peer=" << peer);
PTO_HOST_LOG_INFO("WORKSPACE", "init success");
PTO_HOST_LOG_WARN("SDMA", "fallback=" << fallback);
PTO_HOST_LOG_ERROR("URMA", "failed, ret=" << ret);
```

Logger 构造时读取 `PTO_ISA_GLOBAL_LOG_LEVEL`，空值或非法值使用 ERROR；setter 每次重读环境变量，合法环境值优先于 API 入参，否则使用入参。最终级别非法时返回错误并保留原配置。`FATAL_LEVEL` 仅为关闭阈值，无 FATAL 宏或终止进程行为；直接提交 DEBUG～ERROR 之外的级别也不会进入 callback。

callback 仅允许注册非空函数；重复注册同一函数成功，换成其他函数返回已注册错误，不提供替换或注销。消息指针只在回调期间有效；回调可并发执行，异常直接向调用方传播。

模块传非空字符串，宏支持流式正文和 `std::hex` / `std::dec`。宏先通过 `ShouldLog` 判断是否输出，再构造正文；被过滤时不求值模块及正文表达式，输出时各求值一次。注册 callback 后，四种有效级别均会构造正文并交给 callback。

### 3. 调用流程与格式

`Instance()` 使用 inline 函数内静态对象，同一链接目标的多个翻译单元共享实例；初始化由 C++ 保证线程安全。首次日志或配置调用构造 Logger；文件模式同时构造 FileSink 并准备目录，文件仍延迟打开。stdout 模式不构造 FileSink，stdout 开关只在构造时读取。

调用流程：宏通过 `ShouldLog` 检查级别合法性与出口规则 → 拼接模块 / 源码位置 / 正文 → `Log(LogLevel, const std::string&)` 复用该判断 → 有 callback 则回调并返回，否则格式化后写 stdout 或 FileSink。

文件与 stdout 共用固定格式，无格式模板配置：

```text
<YYYY-MM-DD HH:MM:SS.uuuuuu> <LEVEL> <tid> pid[<pid>] [PTO_HOST <module> <file>:<line>] <message>
2026-10-20 10:44:25.893014 INFO 783415 pid[783415] [PTO_HOST URMA urma_workspace_extract.hpp:55] UrmaInfo OK rank=1 notifyPoolBytes=128
```

时间为本地时间、6 位微秒，级别大写，文件名取 basename，行号取宏调用位置，末尾追加换行。callback 只接收独立级别和 `[PTO_HOST ...] 正文`，不附加时间、级别、tid、pid 头。FileSink 内部故障以 `[PTO_HOST LOGGER]` 直接写 stdout，不经过 Logger，避免递归。

### 4. FileSink 写入与资源管理

1. **目录初始化**：跳过空值、长度达到 `PATH_MAX` 或含任意 `..` 子串的候选路径。相对路径按初始化时 cwd 转绝对路径，规范化重复斜杠及 `.` 后固定。目录创建权限请求 `0750`；getcwd 或选定目录创建失败后禁用该 sink。
2. **写入前处理**：文件 mutex 覆盖整条写入流程。完整记录超过 20 MiB 时截断并附 ` [TRUNCATED]\n`；stdout / callback 不截断。当前文件非空且写入后将超限时关闭并滚动，恰好等于上限允许写入。
3. **新建文件**：按 `statvfs.f_frsize * f_bavail` 检查空间，再老化、独占创建文件。使用 `O_WRONLY | O_CREAT | O_EXCL`，权限请求 `0640`；遇同名文件递增序号重试，序号不因滚动而重置。权限受 umask 影响。
4. **老化规则**：仅扫描命名匹配的普通 PTO 文件，跳过符号链接和目录；按时间戳、PID 数值、序号数值升序删除，为新文件预留一个位置。删除失败不阻止创建；可能删除其他进程仍打开的文件，清理无跨进程强保证。
5. **写入与关闭**：每条调用一次 `write()`，不使用 `ofstream`、不逐条 flush 或 fsync。写入失败 / 短写时报告并关闭，当前记录不补写，下一条可重试建文件；关闭时先清空内部状态，再尝试 `fchmod(0440)` 和 `close()`。
