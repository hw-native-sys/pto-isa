# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

.global  _start
  .type   _start,@function
  .text
_start:
  bstart.std call main
  # C.SETRET encodes halfwords: +1 reaches the next 2-byte instruction.
  c.setret 2, ->ra
_end:
  # ACRC is a SYS-only operation under the current PTO ISA.
  c.bstart.sys fall
  addi zero, 0x5e, ->x1
  acrc 1
  c.bstop
