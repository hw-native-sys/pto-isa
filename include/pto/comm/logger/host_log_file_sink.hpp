/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_HOST_LOG_FILE_SINK_H_
#define PTO_COMM_HOST_LOG_FILE_SINK_H_

#ifdef __CCE_KT_TEST__
#error "PTO Host logging is not supported in Device code."
#endif

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace pto::comm::hostlog {

class PtoHostLogFileSink {
public:
    PtoHostLogFileSink() { InitLogDirectory(); }

    ~PtoHostLogFileSink() { CloseFile(); }

    PtoHostLogFileSink(const PtoHostLogFileSink&) = delete;
    PtoHostLogFileSink& operator=(const PtoHostLogFileSink&) = delete;
    PtoHostLogFileSink(PtoHostLogFileSink&&) = delete;
    PtoHostLogFileSink& operator=(PtoHostLogFileSink&&) = delete;

    void WriteLog(const std::string& content)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_) {
            return;
        }

        const std::string* record = &content;
        std::string truncated;
        if (content.size() > MAX_LOG_FILE_BYTES) {
            constexpr char MARKER[] = " [TRUNCATED]\n";
            truncated.assign(content, 0, MAX_LOG_FILE_BYTES - (sizeof(MARKER) - 1));
            truncated.append(MARKER);
            record = &truncated;
        }
        if (fileSize_ != 0 && record->size() > MAX_LOG_FILE_BYTES - fileSize_) {
            CloseFile();
        }
        if (fd_ < 0 && !OpenNewFile()) {
            return;
        }

        const ssize_t written = write(fd_, record->data(), record->size());
        if (written < 0 || static_cast<size_t>(written) != record->size()) {
            const int savedErrno = written < 0 ? errno : 0;
            std::fprintf(
                stdout, "[PTO_HOST LOGGER] write failed: expected=%zu actual=%zd errno=%d\n", record->size(), written,
                savedErrno);
            CloseFile();
            return;
        }
        fileSize_ += static_cast<size_t>(written);
    }

private:
    static constexpr size_t MAX_LOG_FILE_BYTES = 20 * 1024 * 1024;
    static constexpr size_t MAX_LOG_FILE_COUNT = 64;
    static constexpr uintmax_t MIN_FREE_BYTES = uintmax_t{10} * 1024 * 1024 * 1024;

    struct LogFile {
        std::string name;
        std::string timestamp;
        std::string pid;
        std::string sequence;
    };

    struct DirectoryCloser {
        void operator()(DIR* directory) const { closedir(directory); }
    };

    static void ReportError(const char* operation, const std::string& path, int error)
    {
        std::fprintf(stdout, "[PTO_HOST LOGGER] %s failed: path=%s errno=%d\n", operation, path.c_str(), error);
    }

    static bool IsValidPath(const std::string& path)
    {
        return !path.empty() && path.size() < PATH_MAX && path.find("..") == std::string::npos;
    }

    static std::string NormalizePath(const std::string& path)
    {
        std::string result;
        size_t start = 0;
        while (start < path.size()) {
            const size_t end = path.find('/', start);
            const std::string component = path.substr(start, end == std::string::npos ? end : end - start);
            if (!component.empty() && component != ".") {
                result.push_back('/');
                result += component;
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
        return result.empty() ? "/" : result;
    }

    bool CreateDirectories()
    {
        size_t end = 1;
        while (true) {
            end = directory_.find('/', end);
            const std::string current = directory_.substr(0, end);
            if (mkdir(current.c_str(), S_IRWXU | S_IRGRP | S_IXGRP) != 0 && errno != EEXIST) {
                ReportError("mkdir", current, errno);
                return false;
            }
            struct stat status {};
            if (stat(current.c_str(), &status) != 0) {
                ReportError("stat directory", current, errno);
                return false;
            }
            if (!S_ISDIR(status.st_mode)) {
                ReportError("validate directory", current, ENOTDIR);
                return false;
            }
            if (end == std::string::npos) {
                return true;
            }
            ++end;
        }
    }

    void InitLogDirectory()
    {
        const std::pair<const char*, const char*> candidates[] = {
            {std::getenv("ASCEND_PROCESS_LOG_PATH"), "/debug/plog"},
            {std::getenv("ASCEND_WORK_PATH"), "/log/debug/plog"},
            {std::getenv("HOME"), "/ascend/log/debug/plog"},
            {".", "/ascend/log/debug/plog"},
        };
        for (const auto& candidate : candidates) {
            if (candidate.first == nullptr || candidate.first[0] == '\0') {
                continue;
            }
            std::string path(candidate.first);
            while (path.size() > 1 && path.back() == '/') {
                path.pop_back();
            }
            path += path == "/" ? candidate.second + 1 : candidate.second;
            if (!IsValidPath(path)) {
                continue;
            }
            if (path.front() != '/') {
                char workingDirectory[PATH_MAX];
                if (getcwd(workingDirectory, sizeof(workingDirectory)) == nullptr) {
                    ReportError("getcwd", path, errno);
                    return;
                }
                path = std::string(workingDirectory) + "/" + path;
                if (!IsValidPath(path)) {
                    continue;
                }
            }
            directory_ = NormalizePath(path);
            enabled_ = CreateDirectories();
            return;
        }
        ReportError("select log directory", "", EINVAL);
    }

    bool HasFreeSpace() const
    {
        struct statvfs status {};
        if (statvfs(directory_.c_str(), &status) != 0) {
            ReportError("statvfs", directory_, errno);
            return false;
        }
        const uintmax_t blockBytes = static_cast<uintmax_t>(status.f_frsize);
        const uintmax_t availableBlocks = static_cast<uintmax_t>(status.f_bavail);
        const uintmax_t maximum = std::numeric_limits<uintmax_t>::max();
        const uintmax_t freeBytes =
            blockBytes != 0 && availableBlocks > maximum / blockBytes ? maximum : blockBytes * availableBlocks;
        if (freeBytes <= MIN_FREE_BYTES) {
            std::fprintf(
                stdout, "[PTO_HOST LOGGER] insufficient available disk space: path=%s bytes=%ju\n", directory_.c_str(),
                freeBytes);
            return false;
        }
        return true;
    }

    static bool IsDecimal(const std::string& text)
    {
        return !text.empty() &&
               std::all_of(text.begin(), text.end(), [](char value) { return value >= '0' && value <= '9'; });
    }

    static std::string NormalizeDecimal(const std::string& text)
    {
        const size_t first = text.find_first_not_of('0');
        return first == std::string::npos ? "0" : text.substr(first);
    }

    static bool ParseLogFile(const std::string& name, LogFile& file)
    {
        constexpr char PREFIX[] = "pto_isa_";
        constexpr char SUFFIX[] = ".log";
        constexpr size_t PREFIX_LENGTH = sizeof(PREFIX) - 1;
        constexpr size_t SUFFIX_LENGTH = sizeof(SUFFIX) - 1;
        constexpr size_t MIN_PID_LENGTH = 1;
        constexpr size_t TIMESTAMP_LENGTH = 14;
        constexpr size_t MIN_SEQUENCE_LENGTH = 1;
        constexpr size_t SEPARATOR_LENGTH = 1;
        constexpr size_t MIN_FILENAME_LENGTH = PREFIX_LENGTH + MIN_PID_LENGTH + SEPARATOR_LENGTH + TIMESTAMP_LENGTH +
                                               SEPARATOR_LENGTH + MIN_SEQUENCE_LENGTH + SUFFIX_LENGTH;
        if (name.size() < MIN_FILENAME_LENGTH || name.compare(0, PREFIX_LENGTH, PREFIX) != 0 ||
            name.compare(name.size() - SUFFIX_LENGTH, SUFFIX_LENGTH, SUFFIX) != 0) {
            return false;
        }
        const size_t pidEnd = name.find('_', PREFIX_LENGTH);
        if (pidEnd == std::string::npos) {
            return false;
        }
        const size_t timestampEnd = name.find('_', pidEnd + SEPARATOR_LENGTH);
        if (timestampEnd == std::string::npos || timestampEnd - pidEnd - SEPARATOR_LENGTH != TIMESTAMP_LENGTH) {
            return false;
        }
        const std::string pid = name.substr(PREFIX_LENGTH, pidEnd - PREFIX_LENGTH);
        const std::string timestamp = name.substr(pidEnd + SEPARATOR_LENGTH, TIMESTAMP_LENGTH);
        const std::string sequence =
            name.substr(timestampEnd + SEPARATOR_LENGTH, name.size() - SUFFIX_LENGTH - timestampEnd - SEPARATOR_LENGTH);
        if (!IsDecimal(pid) || !IsDecimal(timestamp) || !IsDecimal(sequence)) {
            return false;
        }
        file = {name, timestamp, NormalizeDecimal(pid), NormalizeDecimal(sequence)};
        return true;
    }

    static bool DecimalLess(const std::string& left, const std::string& right)
    {
        return left.size() == right.size() ? left < right : left.size() < right.size();
    }

    void DeleteOldestFiles()
    {
        std::unique_ptr<DIR, DirectoryCloser> directory(opendir(directory_.c_str()));
        if (!directory) {
            ReportError("opendir", directory_, errno);
            return;
        }
        std::vector<LogFile> files;
        while (true) {
            errno = 0;
            const dirent* entry = readdir(directory.get());
            if (entry == nullptr) {
                if (errno != 0) {
                    ReportError("readdir", directory_, errno);
                }
                break;
            }
            LogFile file;
            if (!ParseLogFile(entry->d_name, file)) {
                continue;
            }
            const std::string path = directory_ + "/" + file.name;
            struct stat status {};
            if (lstat(path.c_str(), &status) != 0 || !S_ISREG(status.st_mode)) {
                continue;
            }
            files.push_back(std::move(file));
        }
        std::stable_sort(files.begin(), files.end(), [](const LogFile& left, const LogFile& right) {
            if (left.timestamp != right.timestamp) {
                return left.timestamp < right.timestamp;
            }
            if (left.pid != right.pid) {
                return DecimalLess(left.pid, right.pid);
            }
            return DecimalLess(left.sequence, right.sequence);
        });
        const size_t deleteCount = files.size() >= MAX_LOG_FILE_COUNT ? files.size() - MAX_LOG_FILE_COUNT + 1 : 0;
        for (size_t i = 0; i < deleteCount; ++i) {
            const std::string path = directory_ + "/" + files[i].name;
            if (unlink(path.c_str()) != 0) {
                ReportError("unlink", path, errno);
            }
        }
    }

    bool OpenNewFile()
    {
        if (!HasFreeSpace()) {
            return false;
        }
        DeleteOldestFiles();

        const std::time_t now = std::time(nullptr);
        std::tm localTime{};
        char timestamp[15];
        if (now == std::time_t{-1} || localtime_r(&now, &localTime) == nullptr ||
            std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", &localTime) == 0) {
            ReportError("format file timestamp", directory_, EINVAL);
            return false;
        }

        while (!sequenceExhausted_) {
            const std::string name =
                "pto_isa_" + std::to_string(getpid()) + "_" + timestamp + "_" + std::to_string(sequence_) + ".log";
            if (sequence_ == std::numeric_limits<uint64_t>::max()) {
                sequenceExhausted_ = true;
            } else {
                ++sequence_;
            }
            const std::string path = directory_ + "/" + name;
            const int newFd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0640);
            if (newFd >= 0) {
                fd_ = newFd;
                fileSize_ = 0;
                return true;
            }
            if (errno != EEXIST) {
                ReportError("open", path, errno);
                return false;
            }
        }
        ReportError("allocate file sequence", directory_, EOVERFLOW);
        return false;
    }

    void CloseFile()
    {
        if (fd_ < 0) {
            return;
        }
        const int oldFd = fd_;
        fd_ = -1;
        fileSize_ = 0;
        if (fchmod(oldFd, S_IRUSR | S_IRGRP) != 0) {
            ReportError("fchmod", directory_, errno);
        }
        if (close(oldFd) != 0) {
            ReportError("close", directory_, errno);
        }
    }

    std::string directory_;
    bool enabled_{false};
    int fd_{-1};
    size_t fileSize_{0};
    uint64_t sequence_{0};
    bool sequenceExhausted_{false};
    std::mutex mutex_;
};

} // namespace pto::comm::hostlog

#endif // PTO_COMM_HOST_LOG_FILE_SINK_H_
