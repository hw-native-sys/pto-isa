
function(pto_gpu_tool NAME)
    add_executable(${NAME} ${NAME}.cu)
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../include
        ${PROJECT_SOURCE_DIR}/../../common
    )
    target_compile_options(${NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O2>
    )
    target_link_libraries(${NAME} PRIVATE Threads::Threads CUDA::cudart)
endfunction()

function(pto_gpu_st NAME)
    add_executable(${NAME} ${NAME}.cu)
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../include
        ${PROJECT_SOURCE_DIR}/../../common
    )
    target_compile_options(${NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O2>
    )
    target_link_libraries(${NAME} PRIVATE Threads::Threads CUDA::cudart)
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

add_subdirectory(pto_gpu_core)
add_subdirectory(pto_gpu_perf)
