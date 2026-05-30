pto_costmodel_sim_st(topk_perf_sim)
target_include_directories(topk_perf_sim PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/topk
)
target_compile_options(topk_perf_sim PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)
