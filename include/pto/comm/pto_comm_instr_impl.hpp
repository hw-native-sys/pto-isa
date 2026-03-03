/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_INSTR_IMPL_HPP
#define PTO_COMM_INSTR_IMPL_HPP

// Native implementation of communication instructions
// Each instruction is implemented directly using Ascend intrinsics
#if defined(__CCE_AICORE__) && !defined(__CPU_SIM)
// Point-to-Point Communication (Synchronous)
#include "pto/comm/TPut.hpp"
#include "pto/comm/TGet.hpp"

// Signal-Based Synchronization
#include "pto/comm/TNotify.hpp"
#include "pto/comm/TWait.hpp"
#include "pto/comm/TTest.hpp"

// Collective Communication
#include "pto/comm/TGather.hpp"
#include "pto/comm/TScatter.hpp"
#include "pto/comm/TBroadCast.hpp"
#include "pto/comm/TReduce.hpp"
#endif

#ifdef __CPU_SIM
// Point-to-Point Communication (Synchronous)
#include "pto/cpu/TPut.hpp"
#include "pto/cpu/TGet.hpp"

// Signal-Based Synchronization
#include "pto/cpu/TNotify.hpp"
#include "pto/cpu/TTest.hpp"
#include "pto/cpu/TWait.hpp"

// Collective Communication
#include "pto/cpu/TReduce.hpp"
#include "pto/cpu/TGather.hpp"
#include "pto/cpu/TScatter.hpp"
#endif

#endif // PTO_COMM_INSTR_IMPL_HPP
