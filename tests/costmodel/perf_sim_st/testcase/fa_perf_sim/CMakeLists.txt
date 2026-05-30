pto_costmodel_sim_st(fa_perf_sim)
target_include_directories(fa_perf_sim PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../kernels/manual/common/flash_atten
)
target_compile_options(fa_perf_sim PRIVATE -D__DAV_C220_CUBE__ -D__DAV_CUBE__ -D__DAV_VEC__)
