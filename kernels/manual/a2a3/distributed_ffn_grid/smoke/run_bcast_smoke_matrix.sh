#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# GridPipe smoke coverage matrix (TBROADCAST / TPOP<GridGroup>, plus the unicast
# time-division handover).
#
# run_bcast_smoke.sh / run_unicast_smoke.sh each run ONE configuration; this runs
# the set that together covers the interface.  Every knob below changes a different
# part of the protocol, so a case that only permutes cosmetics is not in the list:
#
#   group flavour      ROW / COL / SUBRECT, and a SUBRECT that is strictly
#                      INSIDE the mesh (cells outside it must stay no-ops)
#   publishers         one source / EVERY member at the same instant (真·同时 MPSC)
#   rounds             1 / many -- many is what exercises slot reuse and the
#                      producer-side credit (baseline + round*(K-1))
#   ring vs group      SlotCount >= K (no ordering obligation at all) /
#                      SlotCount <  K (the caller must publish in waves) /
#                      SlotCount == 1 (every slot changes hands every tile)
#   ticket batch       n = SlotCount (all publishers of a round in one window) /
#                      n = 1 (grants strictly serialised by basek)
#   scale              5 cells / 24 cells -- 24 concurrent publishers atomic-adding
#                      ONE reserved channel's ready count is the doorbell's real load
#   unicast handover   a channel changing owner while the retiring producer's tiles
#                      are STILL UNDRAINED, with the ring wide enough that the two
#                      turns are disjoint / narrow enough that the new owner must
#                      wait on the credit baseline it was handed at bind time
#
# Each case is a full rebuild (the configuration is compile-time), so the matrix
# takes minutes.  --from/--to run a slice of it, which is what fits a bounded
# task-queue slot; --list prints the plan without building anything.
#
# Usage:
#   ./run_bcast_smoke_matrix.sh [--list] [--from N] [--to N] [--max-spins N]
#
# Exit code is non-zero if any case failed, and the summary names which.

set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 24 cells is the physical AICore count on this part, and every participant of a
# collective must be CO-RESIDENT: a core waiting for a peer that is not scheduled
# yet cannot make progress (the FFN demo solves that with per-phase waves; a smoke
# just stays inside one launch).  So no case launches more than 24 blocks.
RUNNERS=(
  "bcast" "bcast" "bcast" "bcast" "bcast" "bcast" "bcast" "bcast" "bcast" "bcast"
  "unicast" "unicast"
)
NAMES=(
  "row-single"
  "row-single-reuse"
  "col-single"
  "row-allsrc"
  "row-allsrc-serial"
  "row-allsrc-wave"
  "row-allsrc-wave1"
  "subrect-allgather-24"
  "subrect-allgather-24-wave"
  "subrect-inner"
  "unicast-handover"
  "unicast-handover-wrap"
)
DESCS=(
  "ROW 1x5, one source, 1 round -- the plain path"
  "ROW 1x5, one source, 5 rounds over a 2-slot ring -- credit-driven slot reuse"
  "COL 5x1, one source, 2 rounds -- the column arm"
  "ROW 1x5, ALL 5 publish, 3 rounds -- 真·同时 MPSC + per-round slot handover"
  "ROW 1x5, ALL 5 publish, ticket batch 1 -- grants strictly serialised by basek"
  "ROW 1x5, ALL 5 publish, 2-slot ring -- 3 waves per round"
  "ROW 1x5, ALL 5 publish, 1-slot ring, batch 1 -- every tile hands the slot over"
  "SUBRECT 3x8 = 24 cells, ALL publish -- 24-way AllGather, 24 adders on one channel"
  "SUBRECT 3x8 = 24 cells, ALL publish, 6-slot ring -- 4 waves of 6"
  "SUBRECT inside a 3x8 mesh (rows 1-2, cols 2-5), ALL 8 publish -- 16 cells no-op"
  "unicast: channel changes owner with the retiring producer's tiles undrained"
  "unicast: same, ring wraps -- the new owner waits on the relayed credit first"
)
ARGS=(
  ""
  "--rounds 5 --slot-count 2"
  "--span-col 1 --grid-rows 5 --grid-cols 1 --rounds 2"
  "--all-sources 1 --rounds 3"
  "--all-sources 1 --rounds 2 --ticket-batch 1"
  "--all-sources 1 --rounds 2 --slot-count 2"
  "--all-sources 1 --rounds 2 --slot-count 1 --ticket-batch 1"
  "--subrect 1 --grid-rows 3 --grid-cols 8 --all-sources 1"
  "--subrect 1 --grid-rows 3 --grid-cols 8 --all-sources 1 --slot-count 6"
  "--subrect 1 --grid-rows 3 --grid-cols 8 --rect-r0 1 --rect-r1 3 --rect-c0 2 --rect-c1 6 --all-sources 1 --rounds 2"
  "--tiles 2 --slot-count 4"
  "--tiles 2 --slot-count 2"
)

FROM=0
TO=$(( ${#NAMES[@]} - 1 ))
LIST=0
EXTRA=""
while [ $# -gt 0 ]; do
    case "$1" in
        (--list)      LIST=1; shift;;
        (--from)      FROM="$2"; shift 2;;
        (--to)        TO="$2"; shift 2;;
        (--max-spins) EXTRA="${EXTRA} --max-spins $2"; shift 2;;
        (*) echo "[ERROR] unknown option: $1"; exit 2;;
    esac
done

if [ "${LIST}" -eq 1 ]; then
    for i in $(seq 0 $(( ${#NAMES[@]} - 1 ))); do
        printf '%2d  %-26s %s\n' "$i" "${NAMES[$i]}" "${DESCS[$i]}"
    done
    exit 0
fi

FAILED=()
for i in $(seq "${FROM}" "${TO}"); do
    echo ""
    echo "=============================================================="
    echo "[$i] ${NAMES[$i]} -- ${DESCS[$i]}"
    echo "=============================================================="
    if [ "${RUNNERS[$i]}" = "unicast" ]; then
        RUNNER="${SCRIPT_DIR}/run_unicast_smoke.sh"
    else
        RUNNER="${SCRIPT_DIR}/run_bcast_smoke.sh"
    fi
    # shellcheck disable=SC2086
    if bash "${RUNNER}" ${ARGS[$i]} ${EXTRA} 2>&1 | tee /tmp/bcast_matrix_$$.log | grep -E "SUCCESS|FAILED|ERROR|max diff|launch\+sync"; then
        :
    fi
    if ! grep -q "\[SUCCESS\]" /tmp/bcast_matrix_$$.log; then
        FAILED+=("[$i] ${NAMES[$i]}")
    fi
    rm -f /tmp/bcast_matrix_$$.log
done

echo ""
echo "=============================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "[MATRIX PASS] cases ${FROM}..${TO} all green"
    exit 0
fi
echo "[MATRIX FAIL] ${#FAILED[@]} case(s):"
printf '  %s\n' "${FAILED[@]}"
exit 1
