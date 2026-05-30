/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_RECORDER_HPP
#define PTO_PERF_SIM_RECORDER_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pto::perf_sim {

// Thread-local subblock ID, set by LAUNCH_KERNEL before kernel execution.
// Used by PtoRecorder/SyncRecorder to filter Cube vs Vec instructions.
inline thread_local uint32_t current_subblock_id = 0;

using event_t = int;

enum class CvSyncKind : uint8_t
{
    None,
    Push,
    Pop,
    Free,
    Alloc,
};

// ── PipeStage: NPU pipeline stages ──

enum class PipeStage : uint8_t
{
    Scalar,
    MTE2_AIV, // GM/L2 → UB
    MTE2_AIC, // GM/L2 → L1
    MTE3,     // L1 → GM, UB → GM, UB → L1
    MTE1,     // L1 → L0A, L1 → L0B, L1 → UB, L1 → L1, L1 → BT/FB
    Vector,   // VEC compute + UB → UB
    Fixpipe,  // L0C → GM, L0C → L1, L0C → UB
    Matrix,   // CUBE (TMATMUL, TGEMV)
    COUNT,
};

constexpr const char *PipeStageName(PipeStage s)
{
    constexpr const char *names[] = {"Scalar", "MTE2_AIV", "MTE2_AIC", "MTE3", "MTE1", "VEC", "FIXP", "CUBE"};
    return names[static_cast<int>(s)];
}

inline bool IsGMAccessPipe(PipeStage s)
{
    return s == PipeStage::MTE2_AIV || s == PipeStage::MTE2_AIC || s == PipeStage::MTE3 || s == PipeStage::Fixpipe;
}

inline bool IsAICStage(PipeStage s)
{
    return s == PipeStage::MTE2_AIC || s == PipeStage::MTE1 || s == PipeStage::Matrix || s == PipeStage::Fixpipe;
}

inline bool IsAIVStage(PipeStage s)
{
    return s == PipeStage::MTE2_AIV || s == PipeStage::Vector || s == PipeStage::MTE3;
}

// IsCubeSidePipe: hardware pipe IDs belonging to CubeCore (AIC)
inline bool IsCubeSidePipe(int hw_pipe)
{
    return hw_pipe == 2 /*PIPE_MTE1*/ || hw_pipe == 5 /*PIPE_M*/ || hw_pipe == 7 /*PIPE_FIX*/;
}

// ── PTO instruction record ──

struct InstrRecord {
    std::string opcode;    // "TLOAD", "TADD", ...
    std::string dtype;     // "fp16", "fp32", "int8", ...
    std::string tile_args; // stringified tile arguments, e.g. "ub_out, ub_a, ub_b"
    int rows = 0;
    int cols = 0;
    uint64_t estimated_cycles = 0;
    PipeStage stage = PipeStage::Vector;
    static constexpr int MAX_WAIT_EVENTS = 8;
    event_t signal_event = -1;
    event_t wait_events[MAX_WAIT_EVENTS] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int wait_count = 0;
    static constexpr int MAX_SYNC_SIGNALS = 8;
    event_t extra_sync_signals[MAX_SYNC_SIGNALS] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int extra_sync_signal_count = 0;
    uint64_t seq = 0;         // global sequence number for merge ordering
    uint32_t core_id = 0;     // recording core
    uint32_t subblock_id = 0; // VecCore index (0 or 1)
    CvSyncKind cv_kind = CvSyncKind::None;
    uint64_t cv_key = 0;      // logical FIFO identity for TPUSH/TPOP token matching
};

enum class SyncKind : uint8_t
{
    Signal,
    Wait,
    Barrier
};

struct SyncRecord {
    SyncKind kind;
    event_t event_id = -1;
    int src_pipe = -1;
    int dst_pipe = -1;
    bool cross_core = false;
    uint64_t seq = 0;         // global sequence number for merge ordering
    uint32_t core_id = 0;     // recording core
    uint32_t subblock_id = 0; // VecCore index (0 or 1)
};

// ── Shared global sequence counter (preserves ordering across recorders) ──

inline uint64_t SharedNextSeq()
{
    static thread_local uint64_t counter = 0;
    return counter++;
}

template <typename Record>
class CoreRecordStorage {
public:
    static std::vector<std::vector<Record>> &AllCoreRecords()
    {
        static thread_local std::vector<std::vector<Record>> per_core;
        return per_core;
    }

    static std::vector<Record> &GetForCore(uint32_t core_id)
    {
        auto &all = AllCoreRecords();
        if (core_id >= all.size())
            all.resize(core_id + 1);
        return all[core_id];
    }

    static std::vector<Record> &Get()
    {
        return GetForCore(ActiveCore());
    }

    static void SetActiveCore(uint32_t id)
    {
        ActiveCoreRef() = id;
        GetForCore(id);
    }

    static uint32_t ActiveCore()
    {
        return ActiveCoreRef();
    }

    static void Clear()
    {
        AllCoreRecords().clear();
        AllCoreRecords().resize(1);
    }

private:
    static uint32_t &ActiveCoreRef()
    {
        static thread_local uint32_t id = 0;
        return id;
    }
};

// ── PTO instruction recorder (per-core, thread-local) ──

class PtoRecorder : public CoreRecordStorage<InstrRecord> {
public:
    static void Record(InstrRecord r)
    {
        auto sub_id = current_subblock_id;
        if (sub_id > 0 && (IsAICStage(r.stage) || r.stage == PipeStage::Scalar))
            return;
        r.subblock_id = sub_id;
        r.core_id = ActiveCore();
        r.seq = NextSeq();
        Get().push_back(std::move(r));
    }
    static size_t Size()
    {
        return Get().size();
    }

private:
    static uint64_t NextSeq()
    {
        return SharedNextSeq();
    }
};

// ── Sync event recorder (per-core, thread-local) ──

class SyncRecorder : public CoreRecordStorage<SyncRecord> {
public:
    static void Signal(event_t id, int src, int dst, bool cross = false)
    {
        auto sub_id = current_subblock_id;
        if (sub_id > 0) {
            // Skip Cube-side signals (already recorded by subblock_id=0)
            if (IsCubeSidePipe(src))
                return;
            // PIPE_MTE2(3) with Cube dst is MTE2_AIC → skip
            if (src == 3 /*PIPE_MTE2*/ && IsCubeSidePipe(dst))
                return;
        }
        Get().push_back({SyncKind::Signal, id, src, dst, cross, NextSeq(), ActiveCore(), sub_id});
    }
    static void Wait(event_t id, int src, int dst, bool cross = false)
    {
        auto sub_id = current_subblock_id;
        if (sub_id > 0) {
            // Skip Cube-side waits (already recorded by subblock_id=0)
            if (IsCubeSidePipe(dst))
                return;
        }
        Get().push_back({SyncKind::Wait, id, src, dst, cross, NextSeq(), ActiveCore(), sub_id});
    }
    static void Barrier(int pipe)
    {
        Get().push_back({SyncKind::Barrier, -1, pipe, -1, false, NextSeq(), ActiveCore()});
    }

private:
    static uint64_t NextSeq()
    {
        return SharedNextSeq();
    }
};

// ── CV FIFO metadata recorder ──
// TPUSH/TPOP are scalar instructions in perf-sim, but they define the ordering
// between FIFO data producers and consumers. Attach a compact logical FIFO key
// to the next recorded TPUSH/TPOP instruction, then let pipe_model build the
// ready-token dependencies in merged instruction order.

class CvSyncRecorder {
public:
    struct PendingMeta {
        CvSyncKind kind = CvSyncKind::None;
        uint64_t key = 0;
    };

    static void SetPending(CvSyncKind kind, uint64_t key)
    {
        Pending() = {kind, key};
    }

    static void ApplyPending(const char *opcode, InstrRecord &record)
    {
        auto &pending = Pending();
        if (pending.kind == CvSyncKind::None)
            return;
        std::string_view op(opcode);
        bool matches = (pending.kind == CvSyncKind::Push && op == "TPUSH") ||
                       (pending.kind == CvSyncKind::Pop && op == "TPOP") ||
                       (pending.kind == CvSyncKind::Free && op == "TFREE") ||
                       (pending.kind == CvSyncKind::Alloc && op == "TALLOC");
        if (matches) {
            record.cv_kind = pending.kind;
            record.cv_key = pending.key;
        }
        pending = {};
    }

    static void Clear()
    {
        Pending() = {};
    }

private:
    static PendingMeta &Pending()
    {
        static thread_local PendingMeta pending;
        return pending;
    }
};

// Direction encoding for TPipe (matches Direction::DIR_* in fifo.hpp)
enum class TPipeDir : uint8_t
{
    DIR_C2V = 1,  // Cube → Vec
    DIR_V2C = 2,  // Vec → Cube
    DIR_BOTH = 3, // Both directions
    DIR_V2C_CTRL = 4
};

inline uint8_t CvKeyDirection(uint64_t key)
{
    return static_cast<uint8_t>((key >> 8) & 0xffu);
}

inline uint64_t NextCvFifoTypeId()
{
    static thread_local uint64_t next_id = 1;
    return next_id++;
}

template <typename Pipe>
uint64_t CvFifoTypeId()
{
    static const uint64_t id = NextCvFifoTypeId();
    return id;
}

template <typename Pipe>
uint64_t MakeCvFifoKey()
{
    return (static_cast<uint64_t>(Pipe::DIR_TYPE) << 8) | (CvFifoTypeId<Pipe>() << 16);
}

// ── Compile-time tile traits (duck typing for pto::Tile<...>) ──

template <typename T>
struct TileTraits {
    static constexpr int rows = 0;
    static constexpr int cols = 0;
    static constexpr const char *dtype_str()
    {
        return "unknown";
    }
};

template <typename T>
concept HasTileDims = requires {
    typename T::DType;
    {
        T::Rows
    } -> std::convertible_to<int>;
    {
        T::Cols
    } -> std::convertible_to<int>;
};

template <HasTileDims T>
struct TileTraits<T> {
    using DType = typename T::DType;
    static constexpr int rows = T::Rows;
    static constexpr int cols = T::Cols;

    static constexpr const char *dtype_str()
    {
        if constexpr (std::is_same_v<DType, float>)
            return "fp32";
        else if constexpr (std::is_same_v<DType, int32_t>)
            return "int32";
        else if constexpr (std::is_same_v<DType, int8_t>)
            return "int8";
        else
            return "fp16";
    }
};

} // namespace pto::perf_sim

#endif
