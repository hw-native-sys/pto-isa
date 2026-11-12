pto_comm_st(tput_async_urma_gather)

# This target exercises the SIMT gather path, so opt into the SIMT declarations
# explicitly; the common public headers no longer pull them in automatically.
target_compile_definitions(tput_async_urma_gather_kernel PRIVATE PTO_URMA_GATHER_ENABLE_SIMT)

if(PTO_URMA_GATHER_FORCE_SCALAR)
    target_compile_definitions(tput_async_urma_gather_kernel PRIVATE PTO_URMA_GATHER_FORCE_SCALAR)
elseif(PTO_URMA_GATHER_FORCE_SIMT)
    target_compile_definitions(tput_async_urma_gather_kernel PRIVATE PTO_URMA_GATHER_FORCE_SIMT)
endif()
