pto_costmodel_sim_st(conv2d_perf_sim)
target_include_directories(conv2d_perf_sim PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../kernels/manual/a2a3/conv2d_forward
)
target_compile_options(conv2d_perf_sim PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)
