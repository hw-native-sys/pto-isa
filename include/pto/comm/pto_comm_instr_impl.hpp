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
#if defined(__CCE_AICORE__) && !(defined(__CPU_SIM) || defined(__COSTMODEL))

#ifdef PTO_NPU_ARCH_A2A3
// Point-to-Point Communication (Synchronous)
#include "pto/comm/a2a3/TPut.hpp"
#include "pto/comm/a2a3/TGet.hpp"
// Point-to-Point Communication (Asynchronous)
#include "pto/comm/a2a3/async/TPutAsync.hpp"
#include "pto/comm/a2a3/async/TPutAsyncBatch.hpp"
#include "../../../pkg_inc/pto/comm/a2a3/async/TPutAsyncNotify.hpp"
#include "pto/comm/a2a3/async/TGetAsync.hpp"
// Signal-Based Synchronization
#include "pto/comm/a2a3/TNotify.hpp"
#include "pto/comm/a2a3/TWait.hpp"
#include "pto/comm/a2a3/TTest.hpp"
// Collective Communication
#include "pto/comm/a2a3/TGather.hpp"
#include "pto/comm/a2a3/TScatter.hpp"
#include "pto/comm/a2a3/TBroadCast.hpp"
#include "pto/comm/a2a3/TReduce.hpp"
#endif

#ifdef PTO_NPU_ARCH_A5
// Point-to-Point Communication (Synchronous)
#include "pto/comm/a5/TPut.hpp"
#include "pto/comm/a5/TGet.hpp"
// Point-to-Point Communication (Asynchronous)
#include "pto/comm/a5/async/TPutAsync.hpp"
#include "pto/comm/a5/async/TPutAsyncBatch.hpp"
#include "../../../pkg_inc/pto/comm/a5/async/TPutAsyncNotify.hpp"
#include "pto/comm/a5/async/TGetAsync.hpp"
// Signal-Based Synchronization
#include "pto/comm/a5/TNotify.hpp"
#include "pto/comm/a5/TWait.hpp"
#include "pto/comm/a5/TTest.hpp"
// Collective Communication
#include "pto/comm/a5/TGather.hpp"
#include "pto/comm/a5/TScatter.hpp"
#include "pto/comm/a5/TBroadCast.hpp"
#include "pto/comm/a5/TReduce.hpp"
#endif

#endif

#ifdef __CPU_SIM
// Point-to-Point Communication (Synchronous)
#include "pto/cpu/comm/TPut.hpp"
#include "pto/cpu/comm/TGet.hpp"

// Signal-Based Synchronization
#include "pto/cpu/comm/TNotify.hpp"
#include "pto/cpu/comm/TTest.hpp"
#include "pto/cpu/comm/TWait.hpp"

// Collective Communication
#include "pto/cpu/comm/TReduce.hpp"
#include "pto/cpu/comm/TGather.hpp"
#include "pto/cpu/comm/TBroadcast.hpp"
#include "pto/cpu/comm/TScatter.hpp"
#endif

#endif // PTO_COMM_INSTR_IMPL_HPP
