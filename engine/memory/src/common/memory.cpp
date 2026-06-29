#include <arc/memory/memory.h>

#include <utility>

namespace arc
{

tracked_memory_resource::tracked_memory_resource(std::string category, std::pmr::memory_resource* upstream)
    : category_(std::move(category))
    , upstream_(upstream ? upstream : std::pmr::get_default_resource())
{
}

std::string_view tracked_memory_resource::category() const noexcept
{
    return category_;
}

memory_stats tracked_memory_resource::stats() const noexcept
{
    return {
        .allocation_count = allocation_count_.load(),
        .deallocation_count = deallocation_count_.load(),
        .bytes_allocated = bytes_allocated_.load(),
        .bytes_deallocated = bytes_deallocated_.load(),
        .bytes_outstanding = bytes_outstanding_.load(),
        .peak_bytes_outstanding = peak_bytes_outstanding_.load()
    };
}

void tracked_memory_resource::reset_stats() noexcept
{
    allocation_count_.store(0);
    deallocation_count_.store(0);
    bytes_allocated_.store(0);
    bytes_deallocated_.store(0);
    bytes_outstanding_.store(0);
    peak_bytes_outstanding_.store(0);
}

void* tracked_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment)
{
    void* pointer = upstream_->allocate(bytes, alignment);
    allocation_count_.fetch_add(1);
    bytes_allocated_.fetch_add(bytes);
    const auto outstanding = bytes_outstanding_.fetch_add(bytes) + bytes;
    update_peak(outstanding);
    return pointer;
}

void tracked_memory_resource::do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment)
{
    upstream_->deallocate(pointer, bytes, alignment);
    deallocation_count_.fetch_add(1);
    bytes_deallocated_.fetch_add(bytes);
    bytes_outstanding_.fetch_sub(bytes);
}

bool tracked_memory_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept
{
    return this == &other;
}

void tracked_memory_resource::update_peak(std::size_t outstanding) noexcept
{
    auto peak = peak_bytes_outstanding_.load();
    while (outstanding > peak && !peak_bytes_outstanding_.compare_exchange_weak(peak, outstanding))
    {
    }
}

tracked_memory_resource& default_tracked_memory_resource()
{
    static tracked_memory_resource resource("default");
    return resource;
}

memory_stats default_memory_stats() noexcept
{
    return default_tracked_memory_resource().stats();
}

} // namespace arc
