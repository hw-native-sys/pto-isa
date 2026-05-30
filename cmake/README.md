# cmake/

This directory contains CMake helper modules used by the repository, mainly for packaging and third-party dependency integration.

## What’s Inside

- Packaging-related CMake logic (the `package` target)
- FetchContent integration that pulls the shared CANN engineering cmake repo
- Third-party dependency download/integration scripts under `cmake/third_party/`

## Key Files

- `cmake/package.cmake`: Packaging entry functions included by the top-level `CMakeLists.txt`; delegates the cpack configuration to `set_cann_cpack_config` provided by the engineering repo
- `cmake/fetch_cann_cmake.cmake`: Fetches `https://gitcode.com/cann/cmake` and exposes its common functions (`init_cann_project`, `set_cann_cpack_config`, the bundled `makeself`/install scripts, etc.)
- `cmake/func.cmake`: Project-local helpers (protobuf, signing, packing)
- `cmake/third_party/`: Third-party dependency helper scripts

## Entry Points

- Top-level `CMakeLists.txt` includes `cmake/package.cmake` and invokes the packaging helpers
- `build.sh --pkg` triggers the repository packaging flow
