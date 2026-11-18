/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_LOG_HOST_LOGGER_H_
#define PTO_COMM_LOG_HOST_LOGGER_H_

#if defined(__CCE_KT_TEST__)
#error "host_logger.hpp is a host-only header and cannot be included in device code."
#endif

#if !defined(__linux__)
#error "PTO Host logging currently supports Linux only."
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#include "pto/comm/logger/host_log_file_sink.hpp"

namespace pto::comm::hostlog {

enum LogLevel : int32_t { DEBUG_LEVEL = 0, INFO_LEVEL, WARN_LEVEL, ERROR_LEVEL, FATAL_LEVEL, BUTT_LEVEL };

enum LogStatus : int32_t { LOG_SUCCESS = 0, LOG_INVALID_ARGUMENT = -1, LOG_ALREADY_REGISTERED = -2 };

using ExternalLog = void (*)(int32_t level, const char* message);

class PtoHostLogger {
public:
    static PtoHostLogger& Instance()
    {
        static PtoHostLogger logger;
        return logger;
    }

    PtoHostLogger(const PtoHostLogger&) = delete;
    PtoHostLogger& operator=(const PtoHostLogger&) = delete;
    PtoHostLogger(PtoHostLogger&&) = delete;
    PtoHostLogger& operator=(PtoHostLogger&&) = delete;

    // Configuration must finish before concurrent logging starts.
    int32_t SetLogLevel(int32_t level)
    {
        LogLevel environmentLevel = ERROR_LEVEL;
        const int32_t effectiveLevel = ReadEnvironmentLevel(environmentLevel) ? environmentLevel : level;
        if (effectiveLevel < DEBUG_LEVEL || effectiveLevel >= BUTT_LEVEL) {
            return LOG_INVALID_ARGUMENT;
        }
        level_ = static_cast<LogLevel>(effectiveLevel);
        return LOG_SUCCESS;
    }

    int32_t SetExternalLogger(ExternalLog logger)
    {
        if (logger == nullptr) {
            return LOG_INVALID_ARGUMENT;
        }
        if (externalLog_ != nullptr && externalLog_ != logger) {
            return LOG_ALREADY_REGISTERED;
        }
        externalLog_ = logger;
        return LOG_SUCCESS;
    }

    bool ShouldLog(LogLevel level) const
    {
        // FATAL is a filtering threshold, not an emitted message level.
        return level >= DEBUG_LEVEL && level <= ERROR_LEVEL && (externalLog_ != nullptr || level >= level_);
    }

    void Log(LogLevel level, const std::string& message)
    {
        if (!ShouldLog(level)) {
            return;
        }
        if (externalLog_ != nullptr) {
            externalLog_(static_cast<int32_t>(level), message.c_str());
            return;
        }
        const std::string content = FormatRecord(level, message);
        if (toStdout_) {
            std::cout << content;
        } else {
            fileSink_->WriteLog(content);
        }
    }

private:
    PtoHostLogger()
    {
        (void)ReadEnvironmentLevel(level_);
        const char* stdoutValue = std::getenv("PTO_ISA_SLOG_PRINT_TO_STDOUT");
        toStdout_ = stdoutValue != nullptr && std::strcmp(stdoutValue, "1") == 0;
        if (!toStdout_) {
            // Construct the sink with the singleton; only opening its file is lazy.
            fileSink_ = std::make_unique<PtoHostLogFileSink>();
        }
    }

    ~PtoHostLogger() { externalLog_ = nullptr; }

    static bool ReadEnvironmentLevel(LogLevel& level)
    {
        const char* value = std::getenv("PTO_ISA_GLOBAL_LOG_LEVEL");
        if (value == nullptr || value[0] < '0' || value[0] > '4' || value[1] != '\0') {
            return false;
        }
        // Values 0..3 select a severity; 4 disables our file/stdout output.
        level = static_cast<LogLevel>(value[0] - '0');
        return true;
    }

    static const char* LevelName(LogLevel level)
    {
        switch (level) {
            case DEBUG_LEVEL:
                return "DEBUG";
            case INFO_LEVEL:
                return "INFO";
            case WARN_LEVEL:
                return "WARN";
            case ERROR_LEVEL:
                return "ERROR";
            default:
                return "UNKNOWN";
        }
    }

    static std::string FormatRecord(LogLevel level, const std::string& message)
    {
        constexpr int MICROSECOND_WIDTH = 6;
        timeval now{};
        (void)gettimeofday(&now, nullptr);
        std::tm localTime{};
        (void)localtime_r(&now.tv_sec, &localTime);
        char timestamp[32]{};
        (void)std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &localTime);

        std::ostringstream stream;
        stream << timestamp << '.' << std::setfill('0') << std::setw(MICROSECOND_WIDTH) << now.tv_usec << ' '
               << LevelName(level) << ' ' << syscall(SYS_gettid) << " pid[" << getpid() << "] " << message << '\n';
        return stream.str();
    }

    LogLevel level_{ERROR_LEVEL};
    bool toStdout_{false};
    ExternalLog externalLog_{nullptr};
    std::unique_ptr<PtoHostLogFileSink> fileSink_{nullptr};
};

} // namespace pto::comm::hostlog

#endif // PTO_COMM_LOG_HOST_LOGGER_H_
