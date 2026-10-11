#include "ops/offloaded_sparse_moe/cuda/remote_experts.h"

#include "core/tensor.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif

namespace infernix::ops::offloaded_moe {
namespace {

void require_cuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("remote expert service ") + what + ": " + cudaGetErrorString(error));
    }
}

void relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

// Device scope switch for the calling thread, restored on exit.
class DeviceScope {
public:
    explicit DeviceScope(int device) {
        require_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
        require_cuda(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceScope() { (void)cudaSetDevice(previous_); }
    DeviceScope(const DeviceScope&)            = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_ = 0;
};

// y[:, i] = outputs[:, entry[i]] for i < entries, written to the mapped y (16-byte vectors).
__global__ void gather_kernel(const int4* __restrict__ outputs, const std::int32_t* __restrict__ entry, int entries,
                              int vectors, int4* __restrict__ y) {
    const int items  = entries * vectors;
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    for (int item = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); item < items; item += stride) {
        const int i = item / vectors, v = item % vectors;
        y[item]     = outputs[static_cast<std::size_t>(entry[i]) * vectors + v];
    }
}

// Every 144-byte unit of `records` records into the row-pair layout (MoeExpertSource::row_pair_units):
// code byte 32q + 4i + j moves to 16i + 4q + j; the 16 scale bytes stay.
__global__ void row_pair_kernel(std::uint8_t* __restrict__ records, std::size_t units) {
    const std::size_t u = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (u >= units) { return; }
    std::uint8_t* unit = records + u * kUnitBytes;
    std::uint32_t in[32], out[32];
#pragma unroll
    for (int w = 0; w < 32; ++w) { in[w] = reinterpret_cast<const std::uint32_t*>(unit)[w]; }
    // word 8q + i (bytes 32q + 4i .. + 3) becomes word 4i + q
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { out[4 * i + q] = in[8 * q + i]; }
    }
#pragma unroll
    for (int w = 0; w < 32; ++w) { reinterpret_cast<std::uint32_t*>(unit)[w] = out[w]; }
}

// The dispatch block's layout on the remote GPU (and in its pinned staging): MoeDispatch's arrays
// for E experts and kMaxRemoteEntries entries, then the request's entry list in its own order.
struct DispatchLayout {
    std::size_t counts, offsets, cursor, entries, jobs, job_count, entry_list, total;
};

DispatchLayout dispatch_layout(std::uint32_t experts) {
    DispatchLayout d{};
    std::size_t at = 0;
    // Fixed arrays first, then the call's entries and right after them its entry list, so one copy
    // of (head + 2 * entries) ints uploads a request (a decode call: ~8 KiB, not the whole block).
    d.counts     = at; at += experts;
    d.offsets    = at; at += experts + 1;
    d.cursor     = at; at += experts;
    d.jobs       = at; at += experts;
    d.job_count  = at; at += 1;
    d.entries    = at; at += kMaxRemoteEntries;
    d.entry_list = 0; // d.entries + the call's entries
    d.total      = at + kMaxRemoteEntries;
    return d;
}

} // namespace

RemoteExpertService::RemoteExpertService(Options options) : options_(std::move(options)) {
    if (options_.device < 0 || options_.layers == 0 || options_.experts == 0 || options_.experts > kMaxRemoteJobs ||
        options_.record_stride < kRecordBytes || options_.record_stride % 16 != 0 ||
        options_.scales.size() != options_.layers) {
        throw std::invalid_argument("remote expert service options are incomplete");
    }
    const std::size_t keys = static_cast<std::size_t>(options_.layers) * options_.experts;
    slot_of_.assign(keys, -1);
    const DispatchLayout layout = dispatch_layout(options_.experts);
    dispatch_ints_              = layout.total;
    {
        const DeviceScope scope(options_.device);
        // The service's own buffers first, then the records take what the reserve leaves.
        require_cuda(cudaMalloc(&frames_, keys * sizeof(std::int32_t)), "cudaMalloc frames");
        require_cuda(cudaMemset(frames_, 0xFF, keys * sizeof(std::int32_t)), "cudaMemset frames");
        require_cuda(cudaMalloc(&scales_, keys * sizeof(ExpertScales)), "cudaMalloc scales");
        for (std::uint32_t l = 0; l < options_.layers; ++l) {
            require_cuda(cudaMemcpy(scales_ + static_cast<std::size_t>(l) * options_.experts, options_.scales[l],
                                    options_.experts * sizeof(ExpertScales), cudaMemcpyHostToDevice),
                         "upload scales");
        }
        require_cuda(cudaMalloc(&x_device_, sizeof(std::uint16_t) * kHidden * kMaxRemoteColumns), "cudaMalloc x");
        require_cuda(cudaMalloc(&outputs_, sizeof(std::uint16_t) * kHidden * kMaxRemoteEntries), "cudaMalloc outputs");
        require_cuda(cudaMalloc(&dispatch_, dispatch_ints_ * sizeof(std::int32_t)), "cudaMalloc dispatch");
        const std::size_t workspace = moe_experts_workspace_bytes(static_cast<std::int32_t>(options_.experts),
                                                                  kMaxRemoteEntries, kMaxRemoteColumns,
                                                                  options_.activation);
        require_cuda(cudaMalloc(&workspace_, workspace), "cudaMalloc workspace");
        require_cuda(cudaMalloc(&act_plane_, moe_act_plane_bytes(kMaxRemoteEntries)), "cudaMalloc act plane");
        std::size_t free = 0, total = 0;
        require_cuda(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
        const std::uint64_t usable = free > options_.reserve_bytes ? free - options_.reserve_bytes : 0;
        std::uint64_t fit          = usable / options_.record_stride;
        fit                        = std::min<std::uint64_t>(fit, keys);
        if (options_.max_records != 0) { fit = std::min<std::uint64_t>(fit, options_.max_records); }
        capacity_ = static_cast<std::uint32_t>(fit);
        if (capacity_ > 0) {
            require_cuda(cudaMalloc(&records_, static_cast<std::size_t>(capacity_) * options_.record_stride),
                         "cudaMalloc records");
        }
    }
    {
        const DeviceScope scope(options_.main_device);
        require_cuda(cudaMalloc(&held_tables_, keys * sizeof(std::int32_t)), "cudaMalloc held tables");
        require_cuda(cudaMemset(held_tables_, 0xFF, keys * sizeof(std::int32_t)), "cudaMemset held tables");
        require_cuda(cudaMalloc(&call_, sizeof(RemoteCall)), "cudaMalloc call");
        require_cuda(cudaMemset(call_, 0, sizeof(RemoteCall)), "cudaMemset call");
        require_cuda(cudaMalloc(&sequence_, sizeof(std::uint32_t)), "cudaMalloc sequence");
        require_cuda(cudaMemset(sequence_, 0, sizeof(std::uint32_t)), "cudaMemset sequence");
    }
    const auto mapped = [](std::size_t bytes) {
        void* p = nullptr;
        require_cuda(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
        std::memset(p, 0, bytes);
        return p;
    };
    request_ = static_cast<RemoteRequest*>(mapped(sizeof(RemoteRequest)));
    x_       = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * kHidden * kMaxRemoteColumns));
    y_       = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * kHidden * kMaxRemoteEntries));
    done_    = static_cast<std::uint32_t*>(mapped(64));
    void* staging = nullptr;
    require_cuda(cudaHostAlloc(&staging, dispatch_ints_ * sizeof(std::int32_t), cudaHostAllocPortable),
                 "cudaHostAlloc staging");
    staging_ = static_cast<std::int32_t*>(staging);
}

RemoteExpertService::~RemoteExpertService() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) { thread_.join(); }
    cudaFreeHost(staging_);
    cudaFreeHost(done_);
    cudaFreeHost(y_);
    cudaFreeHost(x_);
    cudaFreeHost(request_);
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess) {
        if (cudaSetDevice(options_.main_device) == cudaSuccess) {
            cudaFree(sequence_);
            cudaFree(call_);
            cudaFree(held_tables_);
        }
        if (cudaSetDevice(options_.device) == cudaSuccess) {
            cudaFree(records_);
            cudaFree(act_plane_);
            cudaFree(workspace_);
            cudaFree(dispatch_);
            cudaFree(outputs_);
            cudaFree(x_device_);
            cudaFree(scales_);
            cudaFree(frames_);
        }
        (void)cudaSetDevice(previous);
    }
}

void RemoteExpertService::load(std::span<const std::uint32_t> keys, const Read& read) {
    if (thread_.joinable()) { throw std::logic_error("remote expert service: load() after serving started"); }
    const std::size_t total = static_cast<std::size_t>(options_.layers) * options_.experts;
    const std::size_t n     = std::min<std::size_t>(keys.size(), capacity_);
    constexpr std::size_t kBatch = 32;
    void* pinned                 = nullptr;
    require_cuda(cudaHostAlloc(&pinned, kBatch * options_.record_stride, cudaHostAllocPortable), "cudaHostAlloc load");
    std::vector<std::int32_t> table(total, -1);
    {
        const DeviceScope scope(options_.device);
        auto* bytes = static_cast<std::uint8_t*>(pinned);
        for (std::size_t base = 0; base < n; base += kBatch) {
            const std::size_t count = std::min(kBatch, n - base);
            for (std::size_t i = 0; i < count; ++i) {
                const std::uint32_t key = keys[base + i];
                if (key >= total || slot_of_[key] >= 0) { throw std::invalid_argument("remote expert service: bad key"); }
                read(key, bytes + i * options_.record_stride);
            }
            require_cuda(cudaMemcpy(records_ + base * options_.record_stride, bytes, count * options_.record_stride,
                                    cudaMemcpyHostToDevice),
                         "upload records");
            if (options_.row_pair_units) {
                static_assert(kRecordBytes % kUnitBytes == 0);
                const std::size_t units = count * options_.record_stride / kUnitBytes;
                row_pair_kernel<<<static_cast<unsigned>((units + 255) / 256), 256>>>(
                    records_ + base * options_.record_stride, units);
                require_cuda(cudaGetLastError(), "row pair layout");
            }
            for (std::size_t i = 0; i < count; ++i) {
                const std::uint32_t key = keys[base + i];
                slot_of_[key]           = static_cast<std::int32_t>(base + i);
                table[key]              = static_cast<std::int32_t>(base + i);
            }
        }
        require_cuda(cudaMemcpy(frames_, table.data(), total * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                     "upload frame tables");
    }
    cudaFreeHost(pinned);
    held_ = static_cast<std::uint32_t>(n);
    {
        const DeviceScope scope(options_.main_device);
        require_cuda(cudaMemcpy(held_tables_, table.data(), total * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                     "upload held tables");
    }
    thread_ = std::thread([this] { serve(); });
}

MoeRemoteChannel RemoteExpertService::channel(int layer) const {
    MoeRemoteChannel out; // the held tables hold -1 until load(): no remote jobs before it
    out.held      = held_tables_ + static_cast<std::size_t>(layer) * options_.experts;
    out.request   = request_;
    out.x         = x_;
    out.y         = y_;
    out.done      = done_;
    out.status    = done_ + 1;
    out.heartbeat = done_ + 2;
    out.sequence  = sequence_;
    out.call      = call_;
    out.layer     = layer;
    return out;
}

RemoteExpertService::Stats RemoteExpertService::stats() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

std::string RemoteExpertService::failure() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return failure_;
}

void RemoteExpertService::record_failure(std::string what) noexcept {
    try {
        const std::lock_guard<std::mutex> lock(mutex_);
        failure_ = what.empty() ? std::string("unknown exception") : std::move(what);
        std::fprintf(stderr, "remote expert service stopped: %s\n", failure_.c_str());
    } catch (...) {}
}

void RemoteExpertService::serve() {
    try {
        require_cuda(cudaSetDevice(options_.device), "cudaSetDevice");
        cudaStream_t stream = nullptr;
        require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        cudaEvent_t marks[4] = {};
        for (auto& mark : marks) { require_cuda(cudaEventCreate(&mark), "cudaEventCreate"); }
        const std::uint32_t E      = options_.experts;
        const DispatchLayout d     = dispatch_layout(E);
        auto* volatile_sequence    = reinterpret_cast<volatile std::uint32_t*>(&request_->sequence);
        auto* volatile_done        = reinterpret_cast<volatile std::uint32_t*>(done_);
        auto* volatile_status      = reinterpret_cast<volatile std::uint32_t*>(done_ + 1);
        auto* volatile_beat        = reinterpret_cast<volatile std::uint32_t*>(done_ + 2);
        std::uint32_t beat         = 0;
        std::uint32_t seen         = 0; // sequences are never 0: a request written before this thread starts is still new
        auto last                  = std::chrono::steady_clock::now();
        std::uint32_t polls        = 0;
        std::vector<int> order;
        order.reserve(kMaxRemoteJobs);
        while (!stop_.load(std::memory_order_acquire)) {
            const std::uint32_t sequence = *volatile_sequence;
            *volatile_beat = ++beat;
            if (sequence == seen) {
                relax();
                // Spin while requests come; after 20 ms without one, poll every 50 us.
                if (++polls % 4096 == 0 && std::chrono::steady_clock::now() - last > std::chrono::milliseconds(20)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            seen             = sequence;
            const auto start = std::chrono::steady_clock::now();
            auto enqueued          = start;
            const RemoteRequest& r = *request_;
            const int layer = r.layer, columns = r.columns, top_k = r.top_k, jobs = r.jobs, entries = r.entries;
            std::uint32_t status = 0;
            if (layer < 0 || static_cast<std::uint32_t>(layer) >= options_.layers || columns <= 0 ||
                columns > kMaxRemoteColumns || top_k <= 0 || jobs <= 0 || jobs > static_cast<int>(E) || entries <= 0 ||
                entries > kMaxRemoteEntries || static_cast<std::int64_t>(columns) * top_k > kMaxRemoteEntries) {
                status = EINVAL;
            } else {
                // The dispatch of the held jobs, built here: jobs in ascending expert id (as
                // moe_dispatch lists them), each expert's entries at its offset.
                std::int32_t* s = staging_;
                std::fill(s + d.counts, s + d.counts + E, 0);
                order.resize(static_cast<std::size_t>(jobs));
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return r.expert[a] < r.expert[b]; });
                for (int j = 0; j < jobs; ++j) {
                    const int e = r.expert[j];
                    if (e < 0 || static_cast<std::uint32_t>(e) >= E ||
                        slot_of_[static_cast<std::size_t>(layer) * E + static_cast<std::size_t>(e)] < 0) {
                        status = ENOENT;
                    } else {
                        s[d.counts + e] = r.count[j];
                    }
                }
                std::int32_t running = 0;
                for (std::uint32_t e = 0; e < E; ++e) {
                    s[d.offsets + e] = running;
                    running += s[d.counts + e];
                }
                s[d.offsets + E] = running;
                for (int k = 0; k < jobs; ++k) {
                    const int j     = order[static_cast<std::size_t>(k)];
                    const int e     = r.expert[j];
                    s[d.jobs + k]   = e;
                    const int at    = s[d.offsets + e];
                    for (int c = 0; c < r.count[j]; ++c) { s[d.entries + at + c] = r.entry[r.first[j] + c]; }
                }
                s[d.job_count] = jobs;
                const std::size_t list = d.entries + static_cast<std::size_t>(entries);
                std::memcpy(s + list, r.entry, sizeof(std::int32_t) * static_cast<std::size_t>(entries));
                if (status == 0) {
                    require_cuda(cudaEventRecord(marks[0], stream), "event");
                    require_cuda(cudaMemcpyAsync(dispatch_, s, (list + static_cast<std::size_t>(entries)) * sizeof(std::int32_t),
                                                 cudaMemcpyHostToDevice, stream),
                                 "upload dispatch");
                    require_cuda(cudaMemcpyAsync(x_device_, x_, sizeof(std::uint16_t) * kHidden * columns,
                                                 cudaMemcpyHostToDevice, stream),
                                 "upload x");
                    MoeDispatch dispatch;
                    dispatch.counts    = dispatch_ + d.counts;
                    dispatch.offsets   = dispatch_ + d.offsets;
                    dispatch.cursor    = dispatch_ + d.cursor;
                    dispatch.entries   = dispatch_ + d.entries;
                    dispatch.jobs      = dispatch_ + d.jobs;
                    dispatch.job_count = dispatch_ + d.job_count;
                    MoeExpertSource source;
                    source.frame_base    = records_;
                    source.frames        = frames_ + static_cast<std::size_t>(layer) * E;
                    source.host_records  = records_; // never read: every dispatched expert is resident
                    source.record_stride = options_.record_stride;
                    source.scales        = scales_ + static_cast<std::size_t>(layer) * E;
                    source.activation    = options_.activation;
                    source.row_pair_units = options_.row_pair_units;
                    if (options_.fast_call) {
                        source.act_plane    = act_plane_;
                        source.all_resident = true;
                        int widest          = 0;
                        for (int j = 0; j < jobs; ++j) { widest = std::max(widest, r.count[j]); }
                        source.job_columns = widest;
                    }
                    const Tensor x(x_device_, DType::BF16, {kHidden, columns});
                    Tensor outputs(outputs_, DType::BF16, {kHidden, columns * top_k});
                    require_cuda(cudaEventRecord(marks[1], stream), "event");
                    moe_experts(x, dispatch, source, top_k, jobs, workspace_, outputs, stream, true);
                    require_cuda(cudaEventRecord(marks[2], stream), "event");
                    const int vectors = kHidden * 2 / 16;
                    // About one 16-byte vector per thread: the mapped writes of many CTAs are in flight together.
                    const int blocks  = std::min(2048, (entries * vectors + 255) / 256);
                    gather_kernel<<<blocks, 256, 0, stream>>>(reinterpret_cast<const int4*>(outputs_),
                                                             dispatch_ + list, entries, vectors,
                                                             reinterpret_cast<int4*>(y_));
                    require_cuda(cudaGetLastError(), "gather");
                    require_cuda(cudaEventRecord(marks[3], stream), "event");
                    enqueued = std::chrono::steady_clock::now();
                    cudaError_t state = cudaErrorNotReady;
                    while ((state = cudaStreamQuery(stream)) == cudaErrorNotReady) {
                        *volatile_beat = ++beat;
                        relax();
                    }
                    require_cuda(state, "remote call");
                    float upload = 0, compute = 0, gather = 0;
                    (void)cudaEventElapsedTime(&upload, marks[0], marks[1]);
                    (void)cudaEventElapsedTime(&compute, marks[1], marks[2]);
                    (void)cudaEventElapsedTime(&gather, marks[2], marks[3]);
                    const std::lock_guard<std::mutex> lock(mutex_);
                    stats_.upload_ms += upload;
                    stats_.experts_ms += compute;
                    stats_.gather_ms += gather;
                }
            }
            *volatile_status = status;
            std::atomic_thread_fence(std::memory_order_release);
            *volatile_done = sequence;
            const auto end  = std::chrono::steady_clock::now();
            const auto busy = end - start;
            {
                const std::lock_guard<std::mutex> lock(mutex_);
                ++stats_.requests;
                stats_.jobs += static_cast<std::uint64_t>(jobs);
                stats_.entries += static_cast<std::uint64_t>(entries);
                stats_.busy_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(busy).count());
                stats_.enqueue_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(enqueued - start).count());
                stats_.wait_ns += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - enqueued).count());
            }
            last  = std::chrono::steady_clock::now();
            polls = 0;
        }
        for (auto& mark : marks) { cudaEventDestroy(mark); }
        cudaStreamDestroy(stream);
    } catch (const std::exception& error) {
        record_failure(error.what());
    } catch (...) {
        record_failure("unknown exception");
    }
}

} // namespace infernix::ops::offloaded_moe
