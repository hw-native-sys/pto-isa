#!/bin/sh
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# OAT Pre-commit Check Script — Python Edition (oat-py)
# Replaces the Java-binary-based oat_check.sh.
#
# Requires: Python 3.7+  (pip install oat-py>=1.0.0)
# Works on: Linux / macOS / Windows (Git Bash / MSYS2)
#
# Self-healing: strip Windows CRLF if present
_SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
if sed 's/\r//' "$_SCRIPT" 2>/dev/null | diff -q - "$_SCRIPT" >/dev/null 2>&1; then
    :
else
    echo "[OAT] CRLF detected, auto-fixing and re-running..."
    _TMP="${_SCRIPT}.lf"
    sed 's/\r//' "$_SCRIPT" > "$_TMP" && mv "$_TMP" "$_SCRIPT"
    exec sh "$_SCRIPT" "$@"
fi

set -e

# ---------------------------------------------------------------------------
# 0. Locate Python interpreter
# ---------------------------------------------------------------------------
_PYTHON=""
for _candidate in python3 python py; do
    if command -v "$_candidate" >/dev/null 2>&1; then
        _VER=$("$_candidate" -c "import sys; print(sys.version_info >= (3,7))" 2>/dev/null || echo "False")
        if [ "$_VER" = "True" ]; then
            _PYTHON="$_candidate"
            break
        fi
    fi
done

if [ -z "$_PYTHON" ]; then
    echo "[OAT] [WARNING] Python 3.7+ is required but not found. Please install Python 3.7 or later."
    echo "[OAT] Skipping OAT check, continuing commit..."
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. Ensure oat-py is installed
# ---------------------------------------------------------------------------
_OAT_OK=$("$_PYTHON" -c "import importlib.util; print('ok' if importlib.util.find_spec('oat') else 'missing')" 2>/dev/null || echo "missing")
if [ "$_OAT_OK" != "ok" ]; then
    echo "[OAT] oat-py not found. Installing oat-py>=1.0.1 ..."
    "$_PYTHON" -m pip install --quiet "oat-py>=1.0.1"
    _OAT_OK=$("$_PYTHON" -c "import importlib.util; print('ok' if importlib.util.find_spec('oat') else 'missing')" 2>/dev/null || echo "missing")
    if [ "$_OAT_OK" != "ok" ]; then
        echo "[OAT] [WARNING] Failed to install oat-py. Please run: pip install oat-py>=1.0.1"
        echo "[OAT] Skipping OAT check, continuing commit..."
        exit 0
    fi
    echo "[OAT] oat-py installed successfully."
fi

# ---------------------------------------------------------------------------
# 2. Determine repo root and name
# ---------------------------------------------------------------------------
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
REPO_NAME=$(basename "$REPO_ROOT")
OAT_REPORT_DIR="$REPO_ROOT/oat_reports"

echo "[OAT] Running OAT scan (Python Edition) — INCREMENTAL MODE"
echo "[OAT] Project: $REPO_NAME"

# ---------------------------------------------------------------------------
# 3. Collect staged files
# ---------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    # Called directly with file arguments (e.g. manual test)
    FILE_COUNT=$#
    FILE_LIST=""
    for _f in "$@"; do
        # Convert relative paths to absolute
        case "$_f" in
            /*) _abs="$_f" ;;
            *)  _abs="$REPO_ROOT/$_f" ;;
        esac
        [ -f "$_abs" ] || continue
        if [ -z "$FILE_LIST" ]; then
            FILE_LIST="$_abs"
        else
            FILE_LIST="$FILE_LIST,$_abs"
        fi
    done
else
    _STAGED=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null || true)
    if [ -z "$_STAGED" ]; then
        echo "[OAT] No staged files to check. Skipping."
        exit 0
    fi
    FILE_COUNT=$(echo "$_STAGED" | wc -l | tr -d ' ')
    FILE_LIST=""
    for _f in $_STAGED; do
        case "$_f" in
            /*) _abs="$_f" ;;
            *)  _abs="$REPO_ROOT/$_f" ;;
        esac
        [ -f "$_abs" ] || continue
        if [ -z "$FILE_LIST" ]; then
            FILE_LIST="$_abs"
        else
            FILE_LIST="$FILE_LIST,$_abs"
        fi
    done
fi

if [ -z "$FILE_LIST" ]; then
    echo "[OAT] No existing staged files found. Skipping."
    exit 0
fi

echo "[OAT] Checking $FILE_COUNT staged file(s)..."

# ---------------------------------------------------------------------------
# 4. Ensure oat_reports/ exists and is in .gitignore
# ---------------------------------------------------------------------------
mkdir -p "$OAT_REPORT_DIR"

_GITIGNORE="$REPO_ROOT/.gitignore"
for _entry in "oat_reports" "log"; do
    if ! grep -qE "^${_entry}/?$" "$_GITIGNORE" 2>/dev/null; then
        printf "\n%s/\n" "$_entry" >> "$_GITIGNORE" 2>/dev/null || true
        echo "[OAT] Added ${_entry}/ to .gitignore"
    fi
done

# ---------------------------------------------------------------------------
# 5. Build oat command — use OAT.xml if present in repo root
# ---------------------------------------------------------------------------
_OAT_CMD="$_PYTHON -m oat -mode s -s $REPO_ROOT -r $OAT_REPORT_DIR -n $REPO_NAME -w 1 -f $FILE_LIST"

_OAT_XML="$REPO_ROOT/OAT.xml"
if [ -f "$_OAT_XML" ]; then
    _OAT_CMD="$_OAT_CMD -oatconfig $_OAT_XML"
fi

# ---------------------------------------------------------------------------
# 6. Run oat scan
# ---------------------------------------------------------------------------
echo ""
echo "[OAT] Running compliance scan..."

set +e
eval "$_OAT_CMD" >/dev/null 2>&1
_OAT_RC=$?
set -e

if [ "$_OAT_RC" -ne 0 ] && [ "$_OAT_RC" -ne 1 ]; then
    echo ""
    echo "[OAT] [WARNING] oat exited with unexpected code $_OAT_RC."
    echo "[OAT] Try re-running manually:"
    echo "  $_OAT_CMD"
    echo "[OAT] Skipping OAT check, continuing commit..."
    exit 0
fi

# ---------------------------------------------------------------------------
# 7. Parse report and write result.txt
#    Only: Invalid File Type + License Header Invalid (no copyright)
# ---------------------------------------------------------------------------
REPORT_FILE="$OAT_REPORT_DIR/PlainReport_${REPO_NAME}.txt"
RESULT_FILE="$OAT_REPORT_DIR/result.txt"

# Section headers used as stop-boundaries when extracting sections
_ALL_HEADERS="Invalid File Type Total Count:|License Not Compatible Total Count:|License Header Invalid Total Count:|Copyright Header Invalid Total Count:|No License File Total Count:|No Readme.OpenSource Total Count:|No Readme Total Count:|Import Invalid Total Count:|Redundant License File Total Count:|Third Party Software Info Total Count:"

# Extract one section from the report (header line + detail Name: lines)
# Usage: _extract_section <file> <start_marker>
_extract_section() {
    _file="$1"
    _marker="$2"
    awk -v marker="$_marker" -v boundaries="$_ALL_HEADERS" '
        BEGIN { capturing=0 }
        !capturing {
            if (index($0, marker) == 1) { capturing=1; print; next }
            next
        }
        capturing {
            # Check if this line starts a different section header
            n = split(boundaries, hdrs, "|")
            for (i=1; i<=n; i++) {
                if (hdrs[i] != marker && index($0, hdrs[i]) == 1) { exit }
            }
            print
        }
    ' "$_file" | sed '/^[[:space:]]*$/{ N; /^\n$/d }' | awk '
        { lines[NR]=$0 } END { while(NR>0 && lines[NR]~/^[[:space:]]*$/) NR--; for(i=1;i<=NR;i++) print lines[i] }
    '
}

if [ ! -f "$REPORT_FILE" ]; then
    if [ "$_OAT_RC" -eq 0 ]; then
        echo "[OAT] [OK] All checks passed ($FILE_COUNT file(s) checked)."
        exit 0
    else
        echo "[OAT] [WARNING] Report not found: $REPORT_FILE"
        exit 1
    fi
fi

# Parse counts
_INVALID_TYPE=$(grep "^Invalid File Type Total Count:" "$REPORT_FILE" | grep -oE '[0-9]+' | head -1)
_LICENSE_INVALID=$(grep "^License Header Invalid Total Count:" "$REPORT_FILE" | grep -oE '[0-9]+' | head -1)
_INVALID_TYPE=${_INVALID_TYPE:-0}
_LICENSE_INVALID=${_LICENSE_INVALID:-0}

# Extract detail sections
_SECTION_TYPE=$(_extract_section "$REPORT_FILE" "Invalid File Type Total Count:")
_SECTION_LIC=$(_extract_section "$REPORT_FILE" "License Header Invalid Total Count:")

# Fallback to bare count line if section is empty
[ -n "$_SECTION_TYPE" ] || _SECTION_TYPE="Invalid File Type Total Count: $_INVALID_TYPE"
[ -n "$_SECTION_LIC"  ] || _SECTION_LIC="License Header Invalid Total Count: $_LICENSE_INVALID"

# Write result.txt
{
    echo "==================================="
    echo "OAT Scan Result Summary"
    echo "==================================="
    printf "Scan Time: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Project: $REPO_NAME"
    echo "Files Checked: $FILE_COUNT"
    echo ""
    echo "-----------------------------------"
    echo "$_SECTION_TYPE"
    echo ""
    echo "-----------------------------------"
    echo "$_SECTION_LIC"
    echo "==================================="
} > "$RESULT_FILE"

# Clean up full plain report (keep only result.txt)
rm -f "$REPORT_FILE"

# ---------------------------------------------------------------------------
# 8. Block commit if issues found
# ---------------------------------------------------------------------------
TOTAL_ISSUES=$(( _INVALID_TYPE + _LICENSE_INVALID ))

if [ "$TOTAL_ISSUES" -gt 0 ]; then
    echo ""
    echo "===================================================================="
    echo "  OAT: Compliance issues found. Commit blocked."
    echo "===================================================================="
    echo ""
    echo "[OAT] Found $TOTAL_ISSUES compliance issue(s):"
    echo "  - Invalid File Type:       $_INVALID_TYPE"
    echo "  - License Header Invalid:  $_LICENSE_INVALID"
    echo ""
    echo "[OAT] Details:"
    echo "  cat $RESULT_FILE"
    echo ""
    echo "Fix the issues and recommit, or skip with:"
    echo "  git commit --no-verify"
    echo ""
    exit 1
fi

echo ""
echo "[OAT] [OK] All checks passed ($FILE_COUNT file(s) checked)."
echo "[OAT] Summary: cat $RESULT_FILE"
echo ""
exit 0
