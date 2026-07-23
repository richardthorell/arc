#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace arc
{

enum class memory_domain : std::uint8_t
{
    general,
    jobs,
    frame,
    tick,
    world,
    components,
    network,
    assets,
    streaming,
    gpu_upload,
    editor,
    count
};

constexpr std::string_view to_string(memory_domain domain) noexcept
{
    switch (domain)
    {
    case memory_domain::general: return "general";
    case memory_domain::jobs: return "jobs";
    case memory_domain::frame: return "frame";
    case memory_domain::tick: return "tick";
    case memory_domain::world: return "world";
    case memory_domain::components: return "components";
    case memory_domain::network: return "network";
    case memory_domain::assets: return "assets";
    case memory_domain::streaming: return "streaming";
    case memory_domain::gpu_upload: return "gpu_upload";
    case memory_domain::editor: return "editor";
    case memory_domain::count: break;
    }
    return "unknown";
}

struct memory_tag
{
    std::uint32_t id{};
    std::string_view name{ "untagged" };

    friend constexpr bool operator==(memory_tag, memory_tag) noexcept = default;
};

memory_tag make_memory_tag(std::string_view name) noexcept;
memory_tag current_memory_tag() noexcept;

class allocation_tag_scope
{
public:
    explicit allocation_tag_scope(memory_tag tag) noexcept;
    explicit allocation_tag_scope(std::string_view tag) noexcept;
    ~allocation_tag_scope();

    allocation_tag_scope(const allocation_tag_scope&) = delete;
    allocation_tag_scope& operator=(const allocation_tag_scope&) = delete;

private:
    memory_tag previous_{};
};

struct memory_stats
{
    std::size_t allocation_count{};
    std::size_t deallocation_count{};
    std::size_t bytes_allocated{};
    std::size_t bytes_deallocated{};
    std::size_t bytes_outstanding{};
    std::size_t peak_bytes_outstanding{};
};

struct memory_budget
{
    std::size_t soft_limit{};
    std::size_t hard_limit{};
};

struct memory_domain_snapshot
{
    memory_domain domain{ memory_domain::general };
    memory_stats stats{};
    memory_budget budget{};
    bool soft_limit_exceeded{};
};

struct memory_tag_snapshot
{
    memory_tag tag{};
    memory_domain domain{ memory_domain::general };
    std::size_t allocation_count{};
    std::size_t bytes_outstanding{};
};

struct memory_leak_record
{
    std::uintptr_t address{};
    std::size_t bytes{};
    std::size_t alignment{};
    memory_domain domain{ memory_domain::general };
    memory_tag tag{};
    std::uint64_t world_id{};
    std::uint64_t thread_id{};
    std::uint64_t stack_id{};
};

struct memory_snapshot
{
    std::uint64_t sequence{};
    std::size_t physical_memory{};
    std::size_t global_bytes_outstanding{};
    memory_budget global_budget{};
    std::vector<memory_domain_snapshot> domains;
    std::vector<memory_tag_snapshot> tags;
    std::size_t tracked_allocation_count{};
    std::size_t sampled_stack_count{};
    std::size_t pressure_event_count{};
};

struct memory_system_config
{
    std::size_t physical_memory_override{};
    float cpu_soft_budget_fraction{ 0.75f };
    float cpu_hard_budget_fraction{ 0.90f };
    bool track_live_allocations{
#if !defined(NDEBUG)
        true
#else
        false
#endif
    };
    bool capture_call_stacks{
#if !defined(NDEBUG)
        true
#else
        false
#endif
    };
    std::size_t stack_capture_threshold{ 64u * 1024u };
    std::uint32_t small_allocation_sample_rate{ 256 };
    std::uint32_t maximum_stack_frames{ 32 };
};

enum class memory_pressure_level : std::uint8_t
{
    soft,
    hard,
    allocation_failure
};

using memory_pressure_handler = std::function<void(memory_pressure_level, memory_domain, std::size_t)>;

class memory_system
{
public:
    explicit memory_system(
        memory_system_config config = {},
        std::pmr::memory_resource* upstream = std::pmr::new_delete_resource());
    ~memory_system();

    memory_system(const memory_system&) = delete;
    memory_system& operator=(const memory_system&) = delete;

    void* try_allocate(
        std::size_t bytes,
        std::size_t alignment = alignof(std::max_align_t),
        memory_domain domain = memory_domain::general,
        memory_tag tag = {},
        std::uint64_t world_id = 0) noexcept;
    void* allocate(
        std::size_t bytes,
        std::size_t alignment = alignof(std::max_align_t),
        memory_domain domain = memory_domain::general,
        memory_tag tag = {},
        std::uint64_t world_id = 0);
    void deallocate(
        void* pointer,
        std::size_t bytes,
        std::size_t alignment,
        memory_domain domain = memory_domain::general) noexcept;

    void set_budget(memory_domain domain, memory_budget budget);
    memory_budget budget(memory_domain domain) const noexcept;
    memory_budget global_budget() const noexcept;

    std::uint64_t add_pressure_handler(memory_pressure_handler handler);
    bool remove_pressure_handler(std::uint64_t token);

    memory_snapshot snapshot() const;
    std::vector<memory_leak_record> leaks(std::uint64_t world_id = 0) const;
    const memory_system_config& config() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

memory_system& default_memory_system();

class system_memory_resource final : public std::pmr::memory_resource
{
public:
    explicit system_memory_resource(
        memory_system& system,
        memory_domain domain = memory_domain::general,
        memory_tag tag = {},
        std::uint64_t world_id = 0) noexcept;

    memory_domain domain() const noexcept;
    std::uint64_t world_id() const noexcept;

private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

    memory_system* system_{};
    memory_domain domain_{ memory_domain::general };
    memory_tag tag_{};
    std::uint64_t world_id_{};
};

/**
 * Compatibility PMR resource. New systems should prefer system_memory_resource.
 */
class tracked_memory_resource final : public std::pmr::memory_resource
{
public:
    explicit tracked_memory_resource(
        std::string category = "general",
        std::pmr::memory_resource* upstream = std::pmr::get_default_resource());

    std::string_view category() const noexcept;
    memory_stats stats() const noexcept;
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

tracked_memory_resource& default_tracked_memory_resource();
memory_stats default_memory_stats() noexcept;

struct arena_mark
{
    std::size_t block{};
    std::size_t offset{};
    std::uint64_t generation{};
};

/**
 * Block-chained monotonic arena. Growth never relocates previous allocations.
 */
class linear_arena : public std::pmr::memory_resource
{
public:
    explicit linear_arena(
        std::size_t initial_capacity = 256u * 1024u,
        std::pmr::memory_resource* upstream = std::pmr::get_default_resource());
    ~linear_arena() override;

    linear_arena(const linear_arena&) = delete;
    linear_arena& operator=(const linear_arena&) = delete;
    linear_arena(linear_arena&&) noexcept;
    linear_arena& operator=(linear_arena&&) noexcept;

    void* try_allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) noexcept;
    void reset() noexcept;
    arena_mark mark() const noexcept;
    bool rewind(arena_mark value) noexcept;

    std::size_t used() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t peak_used() const noexcept;
    std::uint64_t generation() const noexcept;

private:
    struct block;
    void add_block(std::size_t minimum_capacity);
    void release() noexcept;
    void* do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void*, std::size_t, std::size_t) override;
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

    std::vector<block> blocks_;
    std::pmr::memory_resource* upstream_{};
    std::size_t initial_capacity_{};
    std::size_t active_block_{};
    std::size_t used_{};
    std::size_t peak_used_{};
    std::uint64_t generation_{ 1 };
};

class frame_arena final : public linear_arena
{
public:
    using linear_arena::linear_arena;
};

class tick_arena final : public linear_arena
{
public:
    using linear_arena::linear_arena;
};

/**
 * Thread-safe fixed-block allocator used for packets and other bounded objects.
 */
class fixed_block_pool
{
public:
    explicit fixed_block_pool(
        std::span<const std::size_t> size_classes,
        std::size_t blocks_per_slab = 64,
        std::pmr::memory_resource* upstream = std::pmr::get_default_resource());
    ~fixed_block_pool();

    fixed_block_pool(const fixed_block_pool&) = delete;
    fixed_block_pool& operator=(const fixed_block_pool&) = delete;

    void* try_allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) noexcept;
    void deallocate(void* pointer, std::size_t bytes) noexcept;
    std::size_t pooled_bytes() const noexcept;
    std::size_t outstanding_bytes() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

class network_packet_pool
{
public:
    explicit network_packet_pool(std::pmr::memory_resource* upstream = std::pmr::get_default_resource());

    void* try_allocate(std::size_t bytes) noexcept;
    void deallocate(void* pointer, std::size_t bytes) noexcept;
    std::size_t outstanding_bytes() const noexcept;

private:
    fixed_block_pool pool_;
};

class world_memory_context
{
public:
    explicit world_memory_context(
        memory_system& system = default_memory_system(),
        std::uint64_t world_id = 0,
        memory_budget budget = {});
    ~world_memory_context();

    std::uint64_t world_id() const noexcept;
    std::pmr::memory_resource* world_resource() noexcept;
    std::pmr::memory_resource* component_resource() noexcept;
    std::vector<memory_leak_record> leaks() const;

private:
    memory_system* system_{};
    std::uint64_t world_id_{};
    system_memory_resource world_resource_;
    system_memory_resource component_resource_;
};

/**
 * Bounded, coalescing heap for decoded or streamed asset payloads.
 */
class streaming_heap final : public std::pmr::memory_resource
{
public:
    explicit streaming_heap(
        memory_system& memory,
        std::size_t capacity,
        memory_tag tag = make_memory_tag("assets.streaming"));
    ~streaming_heap() override;

    streaming_heap(const streaming_heap&) = delete;
    streaming_heap& operator=(const streaming_heap&) = delete;

    void* try_allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) noexcept;
    std::size_t capacity() const noexcept;
    std::size_t used() const noexcept;
    std::size_t peak_used() const noexcept;
    std::size_t largest_free_block() const noexcept;

private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace arc
