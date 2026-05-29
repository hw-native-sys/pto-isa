cmake_minimum_required(VERSION 3.16)
project(mhc_kernels)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(-D_FORTIFY_SOURCE=2 -O2 -std=c++17 -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong)
add_link_options(-s -Wl,-z,relro -Wl,-z,now)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS -xc++ "SHELL:-include stdint.h" "SHELL:-include stddef.h")

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# Build all 15 kernel .so files
set(MHC_KERNELS
    expand_to_mhc_fwd expand_to_mhc_bwd
    head_compute_mix_fwd head_compute_mix_bwd
    pre_split_mixes_fwd pre_split_mixes_bwd
    pre_apply_mix_fwd pre_apply_mix_bwd
    pre_norm_fn_fwd
    fn_normw_merge_fwd fn_normw_merge_bwd
    post_fwd post_bwd
    sinkhorn_normalize_fwd sinkhorn_normalize_bwd
)

foreach(KERNEL ${MHC_KERNELS})
    add_library(${KERNEL}_kernel SHARED ${KERNEL}.cpp)
    target_compile_options(${KERNEL}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --npu-arch=dav-3510 -DMEMORY_BASE)
endforeach()

# Build caller .so (wraps expand_to_mhc_fwd for the test)
add_library(mhc_caller SHARED caller.cpp)
target_compile_options(mhc_caller PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --npu-arch=dav-3510 -DMEMORY_BASE)

# Build host test executable
add_executable(mhc_test main.cpp)
target_compile_options(mhc_test PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(mhc_test PRIVATE ${PROJECT_SOURCE_DIR}/../../../../tests/common)
target_link_directories(mhc_test PUBLIC ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(mhc_test PRIVATE mhc_caller ascendcl stdc++ m pthread)
