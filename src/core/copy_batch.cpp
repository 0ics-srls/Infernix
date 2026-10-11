#include "core/copy_batch.h"

#include "core/device.h"

#include <iterator>

namespace infernix {

void CopyBatch::add(void* destination, const void* source, std::size_t bytes) {
    if (bytes == 0) { return; }
    const auto begin = reinterpret_cast<std::uintptr_t>(destination), end = begin + bytes;
    // Overlap: the first pending range starting after `begin` starts before `end`, or the last one
    // starting at or before `begin` ends after it.
    auto next = ranges_.upper_bound(begin);
    const bool overlaps = (next != ranges_.end() && next->first < end) ||
                          (next != ranges_.begin() && std::prev(next)->second > begin);
    if (overlaps) { flush(); }
    destinations_.push_back(destination);
    sources_.push_back(source);
    sizes_.push_back(bytes);
    ranges_.emplace(begin, end);
}

void CopyBatch::flush() {
    if (sizes_.empty()) { return; }
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    std::size_t first         = 0;
#if CUDART_VERSION >= 13000
    CUDA_CHECK(cudaMemcpyBatchAsync(destinations_.data(), sources_.data(), sizes_.data(), sizes_.size(), &attributes,
                                    &first, 1, stream_));
#else
    // CUDA 12.8-12.9 (sm_70/sm_89 builds): the same call with the failure index before the stream.
    std::size_t fail_index = 0;
    CUDA_CHECK(cudaMemcpyBatchAsync(destinations_.data(), const_cast<void**>(sources_.data()), sizes_.data(),
                                    sizes_.size(), &attributes, &first, 1, &fail_index, stream_));
#endif
    destinations_.clear();
    sources_.clear();
    sizes_.clear();
    ranges_.clear();
}

} // namespace infernix
