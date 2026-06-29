#pragma once

#include <atomic>
#include <cstddef>
#include <memory_resource>
#include <string>
#include <string_view>

namespace arc
{

/**
 * @brief Snapshot of allocation counters captured by a tracked resource.
 */
struct memory_stats
{
    std::size_t allocation_count{};
    std::size_t deallocation_count{};
    std::size_t bytes_allocated{};
    std::size_t bytes_deallocated{};
    std::size_t bytes_outstanding{};
    std::size_t peak_bytes_outstanding{};
};

/**
 * @brief `std::pmr::memory_resource` wrapper that records allocation statistics.
 */
class tracked_memory_resource final : public std::pmr::memory_resource
{
public:
    explicit tracked_memory_resource(
        std::string category = "general",
        std::pmr::memory_resource* upstream = std::pmr::get_default_resource());

    /**
     * @brief Return the category assigned to this resource.
     */
    std::string_view category() const noexcept;

    /**
     * @brief Return a point-in-time copy of allocation counters.
     */
    memory_stats stats() const noexcept;

    /**
     * @brief Reset counters without changing the upstream resource.
     */
    void reset_stats() noexcept;

private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

    void update_peak(std::size_t outstanding) noexcept;

    std::string category_;
    std::pmr::memory_resource* upstream_{};
    std::atomic_size_t allocation_count_{};
    std::atomic_size_t deallocation_count_{};
    std::atomic_size_t bytes_allocated_{};
    std::atomic_size_t bytes_deallocated_{};
    std::atomic_size_t bytes_outstanding_{};
    std::atomic_size_t peak_bytes_outstanding_{};
};

/**
 * @brief Return the engine-wide tracked memory resource.
 */
tracked_memory_resource& default_tracked_memory_resource();

/**
 * @brief Return allocation counters from the engine-wide tracked memory resource.
 */
memory_stats default_memory_stats() noexcept;

} // namespace arc
