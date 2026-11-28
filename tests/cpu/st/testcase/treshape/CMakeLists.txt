pto_cpu_sim_st(treshape)

# Compile the NPU implementation separately from the CPU TRESHAPE definition.
# Both target families must accept the header without naming unavailable builtins.
foreach(family IN ITEMS A2A3 A5)
    set(header_target treshape_npu_header_${family})
    add_library(${header_target} OBJECT npu_header.cpp)
    target_include_directories(${header_target} PRIVATE ${PROJECT_SOURCE_DIR}/../../../include)
    target_compile_definitions(${header_target} PRIVATE PTO_CPU_SIM_TARGET_${family})
    add_dependencies(treshape ${header_target})
endforeach()
