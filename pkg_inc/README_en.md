# pkg_inc/

This directory stores **non-external (internal)** headers that are **not exposed**
to upper-layer consumers. It follows the CANN packaging convention:

- **External** headers that form the public API live under `include/`.
- **Internal** headers that are only meant for in-tree modules live under
  `pkg_inc/`.

## Packaging & Installation

`cmake/package.cmake` installs this directory alongside `include/` into
`<arch>-linux/pkg_inc`. The CANN installer already creates the top-level
`pkg_inc -> <arch>-linux/pkg_inc` symlink, so newly added internal headers are
immediately reachable at `<install_path>/pkg_inc/pto/...`.

New internal headers should be added under the matching `pto/` sub-directory
instead of `include/pto/`; headers already placed in `include/` are left as-is.

## Layout

New internal headers follow the same layout as `include/pto/`:

- `pto/comm/`     : communication internals
- `pto/common/`   : platform-independent shared internals
- `pto/costmodel/`: cost model internals
- `pto/cpu/`      : CPU-side internals
- `pto/npu/`      : NPU-side internals (split by SoC generation)
