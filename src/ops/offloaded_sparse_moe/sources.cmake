# Host side of offloaded_sparse_moe (docs/maintainer/qwen3_8-flash-next-design.md §10, §16.2).
# A plain C++ archive: the CPU expert engine needs no CUDA, and its canonical arithmetic must be
# compiled without floating-point contraction or fast-math on every compiler.
add_library(infernix_offloaded_moe_cpu STATIC
  "${CMAKE_CURRENT_LIST_DIR}/cpu/w4a4_expert.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/cpu/w4a16_expert.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/cpu/expert_team.cpp")
infernix_internal_includes(infernix_offloaded_moe_cpu)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(infernix_offloaded_moe_cpu PRIVATE -ffp-contract=off -fno-fast-math)
elseif(MSVC)
  target_compile_options(infernix_offloaded_moe_cpu PRIVATE /fp:precise)
endif()
target_link_libraries(infernix_offloaded_moe_cpu PUBLIC Threads::Threads)
target_link_libraries(infernix_ops PUBLIC infernix_offloaded_moe_cpu)

# The expert kernels in their own non-RDC archive. On the V100 + RTX 4090 pair (CUDA architectures
# "89") it is built for sm_70 and sm_89: the RTX 4090 runs the Program, and the V100's expert service
# runs these same exact kernels on the experts it holds (remote_experts.h), so an expert's output
# has the same bits on either card and on the CPU.
add_library(infernix_offloaded_moe_cuda STATIC
  # GPU narrow route: the same canonical arithmetic, compiled for the device.
  "${CMAKE_CURRENT_LIST_DIR}/cuda/narrow_expert.cu"
  # A whole MoE layer: routing, device-side dispatch, the exact expert kernels and the combine.
  "${CMAKE_CURRENT_LIST_DIR}/cuda/moe_layer.cu"
  # The wide route: experts with more than eight columns on block-scaled tensor cores.
  "${CMAKE_CURRENT_LIST_DIR}/cuda/wide_expert.cu"
  # The second GPU's expert service (the V100): the same kernels on the experts it holds.
  "${CMAKE_CURRENT_LIST_DIR}/cuda/remote_experts.cu")
infernix_internal_includes(infernix_offloaded_moe_cuda)
infernix_cuda_non_rdc_archive(infernix_offloaded_moe_cuda)
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  set_target_properties(infernix_offloaded_moe_cuda PROPERTIES CUDA_ARCHITECTURES "70;89")
endif()
target_link_libraries(infernix_offloaded_moe_cuda PUBLIC infernix_core infernix_offloaded_moe_cpu
                      PRIVATE infernix_cuda_runtime CUDA::cuda_driver)
target_link_libraries(infernix_ops PUBLIC infernix_offloaded_moe_cuda)
# The host side of CPU-served misses: mapped channel buffers and the service thread.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cpu/miss_service.cpp")
# The host side of the SSD tier's fetch channel: mapped request/response words and the responder.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cpu/fetch_channel.cpp")
