/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <utility>
#include <array>
#include <cstddef>
#include <tuple>
#include <functional>

namespace pto_auto {

// MultiStaged Class for executing computations as stages in a pipeline.
// Template parameter: NumStages -> the number of stages in the pipeline
template <int NumStages>
class MultiStaged {
public:
    template <class F>

    AICORE void callWithPragma(F &&f)
    {
#pragma pto v_loop_barrier
        f();
    }

    // run method
    // parameters:
    //     f:   First stage function
    //     fs:  Remaining stage functions
    // Requirement: At least one stage function must be provided.
    // Behavior: Executes the provided stage functions in overlapped/pipelined manner
    template <class F, class... Fs>
    AICORE void run(F &&f, Fs &&...fs)
    {
        constexpr int NumFs = sizeof...(Fs);
        f();
        if constexpr (NumFs > 0) {
            (callWithPragma(fs), ...);
        }
    }
};

// Range Structure: Represents the iteratioln space of a (possibly nested) multibuffered loop
// Template Parameters:
//      Dim0: Dimension of the parent loop
//      Dims: Dimensions of the child loops
template <int Dim0, int... Dims>
struct Range {
    static constexpr std::array<int, 1 + sizeof...(Dims)> dims = {{Dim0, Dims...}};

    static constexpr int getDim(int i)
    {
        return dims[i];
    }

    static constexpr int numDims()
    {
        return 1 + sizeof...(Dims);
    }

    static constexpr bool hasChildDimGTOne()
    {
        if constexpr (sizeof...(Dims) == 0) {
            return false;
        } else {
            return ((Dims > 1) || ...);
        }
    }
};

template <typename T>
struct PopFront;

template <int D0, int D1, int... Dims>
struct PopFront<Range<D0, D1, Dims...>> {
    using type = Range<D1, Dims...>;
};

template <typename T>
using PopFront_t = typename PopFront<T>::type;

// Phases for the MultiBuffered loop execution
enum class Phase
{
    Prologue,
    Main,
    Epilogue
};

// MultiBuffered Loop utility.
// Template Parametrs:
//      NumBuffs:   Number of buffers used for multi-buffering
template <int NumBuffs>
class MultiBuffered {
public:
    // Context object passed to the body - bundles all loop parameters
    template <Phase P, int BufferId>
    struct Context {
        static constexpr Phase phase = P;
        static constexpr int bufferId = BufferId;
        int iter;
    };

    template <class ChildRange>
    // this is for invoking the child loop with the proper pre-feeded range
    class NestedLoopInvoker {
    public:
        MultiBuffered &parent;

        template <int FirstK = 0, int LastK = 0, class Body>
        AICORE void loop(Body &&body)
        {
            parent.template loop<ChildRange, FirstK, LastK>(body);
        }
    };

    // Loop API
    // Template Parameters:
    // 1. Range:    Defines the iteration space of the loop.
    //              A 1D range, represents the iteration count of the non-nested loop.
    //              An ND range, represents the iteration count of a nested loop (parent and its children).
    // 2. FirstK:   Number of iterations executed in the Prologue phase, default = 0.
    // 3. LastK:    Number of iterations executed in the Epilogue phase, default = 0.
    // 4. Body:     Type of the body lambda function.
    // Parameters:  Lambda function exectued for each iteration of the loop.
    template <class Range, int FirstK = 0, int LastK = 0, class Body>
    AICORE void loop(Body &&body)
    {
        constexpr int NumIters = Range::getDim(0);

        if constexpr (Range::numDims() == 1) {
            constexpr bool ShouldUnroll = NumIters != 1;
            loop<NumIters, ShouldUnroll, FirstK, LastK>(body);
        } else {
            constexpr bool ShouldUnroll = Range::getDim(0) > 1 && !Range::hasChildDimGTOne();

            using ChildRange = PopFront_t<Range>;
            NestedLoopInvoker<ChildRange> childInvoker{*this};

            loop<NumIters, ShouldUnroll, FirstK, LastK>([&](auto ctx) { body(ctx, childInvoker); });
        }
    }

    // Loop helper function: executes the loop body in separate phases.
    // The phases are executed sequentially and do not overlap.
    template <int NumIters, bool ShouldUnroll, int FirstK = 0, int LastK = 0, class Body>
    AICORE void loop(Body &&body)
    {
        constexpr int MBFirstK = FirstK;
        constexpr int MBLastK = LastK;
        constexpr int MainK = NumIters - MBFirstK - MBLastK;

        // executing the body in 3 different phases, phases do not overlap.
        if constexpr (FirstK > 0) {
            // prologue phase: from 0 to firstK
            runWIthPhase<MBFirstK, 0, ShouldUnroll, Phase::Prologue>(body);
        }
        if constexpr (MainK > 0) {
            // main phase: from firstK to numIters - lastK
            runWIthPhase<MainK, MBFirstK, ShouldUnroll, Phase::Main>(body);
        }
        if constexpr (LastK > 0) {
            // epilogue phase: from numIters - lastK to numIters
            runWIthPhase<MBLastK, NumIters - MBLastK, ShouldUnroll, Phase::Epilogue>(body);
        }
    }

private:
    template <int NumIters, int Start, bool ShouldUnroll, Phase P, class Body>
    AICORE void runWIthPhase(Body &&body)
    {
        runLoop<NumIters, Start, ShouldUnroll>(
            [&](int i, auto bufferId) { body(Context<P, decltype(bufferId)::value>{i}); });
    }

    template <int buffIndex, class F>
    AICORE void callWithPragma(int i, F &&f)
    {
        if constexpr (buffIndex > 0) {
#pragma pto v_loop_barrier
            f(i * NumBuffs + buffIndex, std::integral_constant<int, buffIndex>{});
        } else {
#pragma pto v_loop_reset
            f(i * NumBuffs + buffIndex, std::integral_constant<int, buffIndex>{});
        }
    }

    template <int NumIters, int Start, bool ShouldUnroll, class F>
    AICORE inline void runLoop(F &&f)
    {
        int i = Start / NumBuffs;

        if constexpr (Start % NumBuffs != 0) {
            unroll_loop(std::make_index_sequence<NumBuffs>{}, [&](auto buffIndex) {
                constexpr int bi = decltype(buffIndex)::value;
                if constexpr (bi >= Start % NumBuffs) {
                    if (i * NumBuffs + bi < Start + NumIters) {
                        callWithPragma<bi>(i, f);
                    }
                }
            });
            ++i;
        }

        if constexpr (ShouldUnroll) {
            constexpr int FullChunkEnd = (Start + NumIters) / NumBuffs;
            for (; i < FullChunkEnd; ++i) {
                unroll_loop(std::make_index_sequence<NumBuffs>{}, [&](auto buffIndex) {
                    constexpr int bi = decltype(buffIndex)::value;
                    callWithPragma<bi>(i, f);
                });
            }
        } else {
            for (; i < Start + NumIters; ++i) {
                f(i, std::integral_constant<int, 0>{});
            }
        }

        // this is for handling the case where NumIters % NumBuffs != 0
        constexpr int Remaining = (Start + NumIters) % NumBuffs;
        if constexpr (Remaining > 0 && ShouldUnroll) {
            unroll_loop(std::make_index_sequence<NumBuffs>{}, [&](auto buffIndex) {
                constexpr int bi = decltype(buffIndex)::value;
                if constexpr (bi < Remaining) {
                    callWithPragma<bi>(i, f);
                }
            });
        }
    }

    template <int K, class BufferId, typename F>
    AICORE auto every(int i, BufferId buffer_id, F &&f)
    {
        constexpr int bi = decltype(buffer_id)::value;
        if constexpr (K == NumBuffs) {
            if constexpr (bi == 0) {
                f();
            }
        } else {
            if (i % K == 0) {
                f();
            }
        }
    }

    template <size_t... Indices, typename F>
    AICORE inline void unroll_loop(std::index_sequence<Indices...>, F &&f)
    {
        (f(std::integral_constant<size_t, Indices>{}), ...);
    }
};

} // namespace pto_auto