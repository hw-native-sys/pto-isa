# PTO-ISA Host Logging

[简体中文](README_zh.md)

## Part I: Requirements and Usage Scenarios

### 1. Scope and Responsibilities

Provide a unified logging interface, log levels, output format, and file management for the PTO-ISA Host side. Callers supply the level, module name, and message body; the logging component handles filtering, formatting, and output.

Logging is synchronous and targets Linux Host environments, including CCU host orchestration. Device logging, `PTO_ASSERT`, and `static_assert` remain unchanged.

### 2. Logging Behavior

Callers choose the log level. Recommended uses are:

| Scenario | Level and behavior |
| --- | --- |
| Parameter validation or API call failure | ERROR; record the cause, return code, and necessary context |
| Continue with a fallback configuration after an abnormal condition | WARN; record the condition and fallback behavior |
| Successful initialization, connection setup, resource release, and stage tracing (Trace) | INFO; record progress and stage timings |
| Additional diagnostic details | DEBUG; enable as needed |

The module name is a caller-supplied string used only as a label, with no predefined list or registration requirement. Each PTO file or stdout record contains the time, level, thread / process ID, module, source location, and message body. Call sites supply application context such as rank, device, and peer.

Three outputs are supported: files, stdout, and an external callback. Once a callback is registered, it is the only output; otherwise, either files or stdout is used. The callback receives all DEBUG / INFO / WARN / ERROR messages submitted to the Logger, bypassing the component's threshold. The receiver performs any filtering.

### 3. Configuration and Paths

| Configuration | Rule |
| --- | --- |
| `PTO_ISA_GLOBAL_LOG_LEVEL` | A single character `0 / 1 / 2 / 3 / 4` selects DEBUG / INFO / WARN / ERROR / disabled file and stdout output; the default is ERROR |
| `PTO_ISA_SLOG_PRINT_TO_STDOUT` | Only the string `1` enables stdout; all other values select files. A callback takes precedence |

The two PTO-prefixed variables independently control PTO logging. Log paths continue to use the ASCEND path variables below; no PTO-specific path or retained-file-count variable is added.

The log directory is selected in the following order. If a candidate variable is unset, empty, or contains an invalid path, the next candidate is tried:

| Priority | Environment variable and meaning | Final directory |
| --- | --- | --- |
| 1 | `ASCEND_PROCESS_LOG_PATH`: optional log root directory | `${ASCEND_PROCESS_LOG_PATH}/debug/plog` |
| 2 | `ASCEND_WORK_PATH`: optional work root directory | `${ASCEND_WORK_PATH}/log/debug/plog` |
| 3 | `HOME`: user home directory, normally set by the system | `${HOME}/ascend/log/debug/plog` |
| 4 | Current working directory when the first three candidates are unavailable | `./ascend/log/debug/plog` |

These variables do not all need to be set. If multiple variables are set, the highest-priority valid value wins. Set each value to a root directory; PTO appends the suffix shown in the table. Path configuration is read and fixed when the Logger is first initialized. Relative paths are resolved against the working directory at that time; later environment or working-directory changes do not change the log directory.

For example, before starting the program:

```bash
export ASCEND_PROCESS_LOG_PATH=/data/pto_logs
```

Logs will be written to `/data/pto_logs/debug/plog/`. The current user needs permission to create directories and write files there. If neither ASCEND path variable is set, `${HOME}/ascend/log/debug/plog` is normally used; if `HOME` is also unavailable, the current working directory is used.

PTO uses separate files under plog and manages writing, rotation, and retention itself.

### 4. File, Concurrency, and Exit Constraints

| Item | Requirement |
| --- | --- |
| Filename | `pto_isa_<pid>_<YYYYMMDDHHMMSS>_<seq>.log`; create exclusively without overwriting existing files |
| Creation time | When the first record is actually written to a file |
| Size and count | 20 MiB per file; a target of 64 matching PTO files across the directory, including the file about to be created; no environment variable controls the count |
| Retention cleanup | When creating a file, remove the oldest files according to timestamps in their names; cleanup is not triggered by age in days or idle time, and files with other prefixes are unaffected |
| Disk space | Before each file creation, the space available to the current user must exceed 10 GiB |

## Part II: Interfaces, Mechanisms, and Flow

### 1. File and Class Responsibilities

Files reside in `include/pto/comm/logger/` and are delivered through the existing include packaging process.

| File | Responsibility |
| --- | --- |
| `host_log.hpp` | Public configuration interfaces, four logging macros, and assembly of the module and source location |
| `host_logger.hpp` | `PtoHostLogger`: levels, callback, formatting, and output selection |
| `host_log_file_sink.hpp` | `PtoHostLogFileSink`: directory, fd, writing, rotation, retention cleanup, and file mutex |

The dependency chain is `host_log.hpp` → `host_logger.hpp` → `host_log_file_sink.hpp`. The Logger exclusively owns its FileSink through `unique_ptr`. The FileSink receives only complete strings and has no knowledge of levels, modules, or callbacks.

Each logging header can be included independently. Inclusion is rejected under `__CCE_KT_TEST__`, and the headers are not included by the Device umbrella header. Callers include `host_log.hpp` outside classes and namespaces.

### 2. Calling Interfaces and Levels

Interface declarations:

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

Usage examples:

```cpp
#include "pto/comm/logger/host_log.hpp"

PTO_HOST_LOG_DEBUG("MY_MODULE", "peer=" << peer);
PTO_HOST_LOG_INFO("WORKSPACE", "init success");
PTO_HOST_LOG_WARN("SDMA", "fallback=" << fallback);
PTO_HOST_LOG_ERROR("URMA", "failed, ret=" << ret);
```

The Logger reads `PTO_ISA_GLOBAL_LOG_LEVEL` during construction and uses ERROR for an empty or invalid value. The setter rereads the environment variable on every call: a valid environment value takes precedence over the API argument; otherwise, the argument is used. If the resulting level is invalid, the setter returns an error and preserves the existing configuration. `FATAL_LEVEL` is only a disabling threshold; there is no FATAL macro or process termination behavior. Levels outside DEBUG–ERROR submitted directly are not delivered to the callback either.

Only a non-null callback function can be registered. Registering the same function again succeeds; registering a different function returns an already-registered error. Replacement and unregistration are not supported. The message pointer is valid only during the callback. Callbacks may execute concurrently, and exceptions propagate directly to the caller.

Pass a non-empty string as the module name. The macros support stream-style message bodies and `std::hex` / `std::dec`. A macro first calls `ShouldLog` to determine whether output is needed, then constructs the message. Filtered messages do not evaluate the module or message expressions; emitted messages evaluate each once. After callback registration, all four valid levels construct their messages and deliver them to the callback.

### 3. Call Flow and Format

`Instance()` uses a static object inside an inline function. Translation units in the same link target share the instance, and C++ guarantees thread-safe initialization. The first logging or configuration call constructs the Logger. File mode also constructs a FileSink and prepares the directory, while opening the file remains deferred. Stdout mode does not construct a FileSink. The stdout switch is read only during construction.

Call flow: the macro checks level validity and output rules through `ShouldLog` → assembles the module / source location / message body → `Log(LogLevel, const std::string&)` reuses that check → invokes the callback and returns if one is registered; otherwise, formats the record and writes it to stdout or the FileSink.

Files and stdout share a fixed format, with no configurable format template:

```text
<YYYY-MM-DD HH:MM:SS.uuuuuu> <LEVEL> <tid> pid[<pid>] [PTO_HOST <module> <file>:<line>] <message>
2026-10-20 10:44:25.893014 INFO 783415 pid[783415] [PTO_HOST URMA urma_workspace_extract.hpp:55] UrmaInfo OK rank=1 notifyPoolBytes=128
```

The timestamp uses local time with six microsecond digits. Levels are uppercase, filenames use the basename, line numbers identify the macro call site, and each record ends with a newline. A callback receives the level separately and the `[PTO_HOST ...] message body`, without the time, level, tid, or pid prefix. FileSink internal failures are written directly to stdout with `[PTO_HOST LOGGER]`, bypassing the Logger to avoid recursion.

### 4. FileSink Writing and Resource Management

1. **Directory initialization**: Skip candidate paths that are empty, reach `PATH_MAX` in length, or contain any `..` substring. Convert relative paths to absolute paths using the cwd at initialization, normalize repeated slashes and `.`, and keep the resulting path fixed. Directory creation requests permissions `0750`. Disable the sink if getcwd or creation of the selected directory fails.
2. **Before writing**: The file mutex covers the entire write operation. Truncate complete records larger than 20 MiB and append ` [TRUNCATED]\n`; stdout / callback output is not truncated. If the current file is nonempty and the write would exceed the limit, close and rotate it. A write that reaches the limit exactly is allowed.
3. **File creation**: Check space using `statvfs.f_frsize * f_bavail`, perform retention cleanup, and create the file exclusively. Use `O_WRONLY | O_CREAT | O_EXCL` and request permissions `0640`. If the name already exists, increment the sequence number and retry; rotation does not reset the sequence number. Permissions are affected by umask.
4. **Retention cleanup**: Scan only regular PTO files whose names match the pattern, skipping symbolic links and directories. Delete in ascending order of timestamp, numeric PID, and numeric sequence number, leaving one slot for the new file. A deletion failure does not prevent creation. Cleanup may delete a file still open in another process and provides no strict cross-process guarantee.
5. **Writing and closing**: Call `write()` once per record; do not use `ofstream` or flush or fsync each record. Report a failed or short write and close the file. Do not retry the current record; the next record may retry file creation. When closing, clear internal state first, then attempt `fchmod(0440)` and `close()`.
