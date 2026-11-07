# Release Notes — PTO Tile Lib

This document summarizes version changes in PTO Tile Lib.

The format follows Keep a Changelog style (Added / Changed / Fixed / Deprecated / Removed / Security).

## Unreleased

- N/A

## 9.1.0 — Initial public release

### Added

- Initial public release of PTO Tile Lib.
- Per-instruction ISA reference pages under `docs/isa/` covering Tile Operation Diagram, Introduction, Math Interpretation, C++ Intrinsic, and Constraints for each primitive.
- CPU simulation path (`tests/run_cpu.py`) and A5 ST path (`tests/script/run_st.py`).

### Changed

- Public fused multiply-add instruction names use the shorter ISA mnemonics: `TFUSEDMULADD` → `TMADD`, `TMULADDDST` → `TMULA`. The legacy names are not retained as public wrappers.
- Ascend 950PR simulator/hardware validation runs recorded against CANN 9.2.0 (see `TEXTRACT`, `TINSERT`, `TMOV`, `TMOV_FP`, `TSTORE`, `TSTORE_FP` ISA pages).

### Fixed

- N/A

### Deprecated

- N/A

### Removed

- Legacy wrapper intrinsics (e.g. `TADDC`, `TSUBC`, `TSUBRELUCONV`, `TSUBSC`, `TSUBVIEW`, `TSYNC`) removed; the compatibility window is closed and no public wrapper is retained. See `docs/isa/README.md` for the full list and migration guidance.

### Security

- For the vulnerability reporting process, see `SECURITY.md`.

## Compatibility Notes

- **Ascend (NPU / simulator)**: requires Ascend CANN toolkit `>= 8.3` (see `version.info`); exact supported SoCs and toolchains depend on the installed CANN distribution.
- **CPU simulator**: intended to run on macOS / Linux / Windows with a C++ toolchain and Python; see `docs/getting-started.md`.
