# Drop-in ST. Same helper as tests/npu/a2a3/comm/st/testcase/tnotify.
# After copy: add this directory name to ALL_TESTCASES immediately after tnotify.
#
# Parent comm/st/CMakeLists.txt injects --cce-pto-enable globally. This kernel
# inlines Set and does not need that flag; if stock bisheng rejects it, the
# whole comm ST tree is already broken for tnotify — fix that first, or use the
# PTO-enabled bisheng that already builds tnotify.
pto_comm_st(tnotify_set_false_share)
