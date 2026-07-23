#include <arc/memory/memory.h>
#include <arc/diagnostics/log.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace arc
{
namespace
{

constexpr std::size_t domain_count = static_cast<std::size_t>(memory_domain::count);
thread_local memory_tag active_tag{};
thread_local bool handling_pressure{};
std::atomic_uint64_t allocation_sequence{};

std::size_t domain_index(memory_domain domain) noexcept
{
    return static_cast<std::size_t>(domain);
}

std::uint32_t fnv1a_32(std::string_view text) noexcept
{
    std::uint32_t value = 2166136261u;
    for (const char character : text)
    {
        value ^= static_cast<std::uint8_t>(character);
        value *= 16777619u;
    }
    return value == 0 ? 1u : value;
}

std::uint64_t current_thread_id() noexcept
{
    return static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

std::size_t physical_memory_bytes() noexcept
{
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? static_cast<std::size_t>(status.ullTotalPhys) : 0;
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0)
        return 0;
    return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
#else
    return 0;
#endif
}

std::uint64_t capture_stack_id(const memory_system_config& config, std::size_t bytes, std::uint64_t sequence) noexcept
{
    if (!config.capture_call_stacks)
        return 0;
    if (bytes < config.stack_capture_threshold &&
        (config.small_allocation_sample_rate == 0 || sequence % config.small_allocation_sample_rate != 0))
        return 0;

#if defined(_WIN32)
    std::array<void*, 32> frames{};
    const auto requested = static_cast<ULONG>(std::min<std::size_t>(config.maximum_stack_frames, frames.size()));
    const USHORT count = CaptureStackBackTrace(2, requested, frames.data(), nullptr);
    std::uint64_t hash = 1469598103934665603ull;
    for (USHORT index = 0; index < count; ++index)
    {
        hash ^= static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(frames[index]));
        hash *= 1099511628211ull;
    }
    return hash == 0 ? 1 : hash;
#else
    (void)config;
    (void)bytes;
    (void)sequence;
    return 0;
#endif
}

bool valid_alignment(std::size_t alignment) noexcept
{
    return alignment != 0 && std::has_single_bit(alignment);
}

}

memory_tag make_memory_tag(std::string_view name) noexcept
{
    return { .id = fnv1a_32(name), .name = name.empty() ? std::string_view("untagged") : name };
}

memory_tag current_memory_tag() noexcept
{
    return active_tag.id == 0 ? make_memory_tag("untagged") : active_tag;
}

allocation_tag_scope::allocation_tag_scope(memory_tag tag) noexcept
    : previous_(active_tag)
{
    active_tag = tag.id == 0 ? make_memory_tag("untagged") : tag;
}

allocation_tag_scope::allocation_tag_scope(std::string_view tag) noexcept
    : allocation_tag_scope(make_memory_tag(tag))
{
}

allocation_tag_scope::~allocation_tag_scope()
{
    active_tag = previous_;
}

struct memory_system::implementation
{
    struct allocation
    {
        std::size_t bytes{};
        std::size_t alignment{};
        memory_domain domain{};
        memory_tag tag{};
        std::uint64_t world_id{};
        std::uint64_t thread_id{};
        std::uint64_t stack_id{};
    };

    explicit implementation(memory_system_config value, std::pmr::memory_resource* resource)
        : config(value)
        , upstream(resource ? resource : std::pmr::new_delete_resource())
    {
        physical_memory = config.physical_memory_override != 0 ? config.physical_memory_override : physical_memory_bytes();
        if (physical_memory != 0)
        {
            global.soft_limit = static_cast<std::size_t>(
                static_cast<long double>(physical_memory) * std::clamp(config.cpu_soft_budget_fraction, 0.0f, 1.0f));
            global.hard_limit = static_cast<std::size_t>(
                static_cast<long double>(physical_memory) * std::clamp(config.cpu_hard_budget_fraction, 0.0f, 1.0f));
        }
    }

    bool over_hard(memory_domain domain, std::size_t bytes) const noexcept
    {
        const auto index = domain_index(domain);
        const auto global_outstanding = outstanding.load(std::memory_order_relaxed);
        if (global.hard_limit != 0 && bytes > global.hard_limit - std::min(global.hard_limit, global_outstanding))
            return true;
        const auto domain_outstanding = stats[index].bytes_outstanding.load(std::memory_order_relaxed);
        return budgets[index].hard_limit != 0 &&
            bytes > budgets[index].hard_limit - std::min(budgets[index].hard_limit, domain_outstanding);
    }

    bool over_soft(memory_domain domain, std::size_t bytes) const noexcept
    {
        const auto index = domain_index(domain);
        const auto global_outstanding = outstanding.load(std::memory_order_relaxed);
        if (global.soft_limit != 0 && bytes > global.soft_limit - std::min(global.soft_limit, global_outstanding))
            return true;
        const auto domain_outstanding = stats[index].bytes_outstanding.load(std::memory_order_relaxed);
        return budgets[index].soft_limit != 0 &&
            bytes > budgets[index].soft_limit - std::min(budgets[index].soft_limit, domain_outstanding);
    }

    struct atomic_stats
    {
        std::atomic_size_t allocation_count{};
        std::atomic_size_t deallocation_count{};
        std::atomic_size_t bytes_allocated{};
        std::atomic_size_t bytes_deallocated{};
        std::atomic_size_t bytes_outstanding{};
        std::atomic_size_t peak_bytes_outstanding{};
    };

    memory_system_config config;
    std::pmr::memory_resource* upstream{};
    std::size_t physical_memory{};
    memory_budget global{};
    std::array<memory_budget, domain_count> budgets{};
    std::array<atomic_stats, domain_count> stats{};
    std::array<std::atomic_bool, domain_count> soft_pressure_announced{};
    std::atomic_size_t outstanding{};
    std::atomic_size_t pressure_events{};
    mutable std::mutex mutex;
    std::unordered_map<void*, allocation> allocations;
    std::unordered_map<std::uint32_t, std::string> tag_names;
    std::unordered_map<std::uint64_t, memory_pressure_handler> handlers;
    std::uint64_t next_handler{ 1 };
    mutable std::atomic_uint64_t snapshot_sequence{};
};

memory_system::memory_system(memory_system_config config, std::pmr::memory_resource* upstream)
    : implementation_(std::make_unique<implementation>(config, upstream))
{
}

memory_system::~memory_system() = default;

void* memory_system::try_allocate(
    std::size_t bytes,
    std::size_t alignment,
    memory_domain domain,
    memory_tag tag,
    std::uint64_t world_id) noexcept
{
    return try_allocate_result(bytes, alignment, domain, tag, world_id).pointer;
}

allocation_result memory_system::try_allocate_result(
    std::size_t bytes,
    std::size_t alignment,
    memory_domain domain,
    memory_tag tag,
    std::uint64_t world_id) noexcept
{
    if (bytes == 0)
        bytes = 1;
    if (!valid_alignment(alignment) || domain == memory_domain::count)
        return { .error = allocation_error::invalid_request };
    if (tag.id == 0)
        tag = current_memory_tag();

    auto run_pressure_handlers = [&](memory_pressure_level level) {
        if (handling_pressure)
            return;
        handling_pressure = true;
        implementation_->pressure_events.fetch_add(1, std::memory_order_relaxed);
        std::vector<memory_pressure_handler> handlers;
        try
        {
            std::lock_guard lock(implementation_->mutex);
            handlers.reserve(implementation_->handlers.size());
            for (const auto& [_, handler] : implementation_->handlers)
                handlers.push_back(handler);
        }
        catch (...)
        {
        }
        for (auto& handler : handlers)
        {
            try
            {
                handler(level, domain, bytes);
            }
            catch (...)
            {
            }
        }
        handling_pressure = false;
    };

    if (implementation_->over_hard(domain, bytes))
    {
        run_pressure_handlers(memory_pressure_level::hard);
        if (implementation_->over_hard(domain, bytes))
            return { .error = allocation_error::budget_exceeded };
    }
    else if (implementation_->over_soft(domain, bytes))
    {
        const auto index = domain_index(domain);
        bool expected = false;
        if (implementation_->soft_pressure_announced[index].compare_exchange_strong(
                expected, true, std::memory_order_relaxed))
        {
            run_pressure_handlers(memory_pressure_level::soft);
        }
    }

    void* pointer{};
    try
    {
        pointer = implementation_->upstream->allocate(bytes, alignment);
    }
    catch (...)
    {
        run_pressure_handlers(memory_pressure_level::allocation_failure);
        try
        {
            pointer = implementation_->upstream->allocate(bytes, alignment);
        }
        catch (...)
        {
            return { .error = allocation_error::upstream_failure };
        }
    }

    const auto sequence = allocation_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto stack_id = capture_stack_id(implementation_->config, bytes, sequence);
    const auto index = domain_index(domain);
    auto& stats = implementation_->stats[index];
    stats.allocation_count.fetch_add(1, std::memory_order_relaxed);
    stats.bytes_allocated.fetch_add(bytes, std::memory_order_relaxed);
    const auto domain_outstanding = stats.bytes_outstanding.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto peak = stats.peak_bytes_outstanding.load(std::memory_order_relaxed);
    while (domain_outstanding > peak &&
        !stats.peak_bytes_outstanding.compare_exchange_weak(peak, domain_outstanding, std::memory_order_relaxed))
    {
    }
    implementation_->outstanding.fetch_add(bytes, std::memory_order_relaxed);

    if (implementation_->config.track_live_allocations)
    {
        try
        {
            std::lock_guard lock(implementation_->mutex);
            implementation_->tag_names.try_emplace(tag.id, tag.name);
            const auto stored = implementation_->tag_names.find(tag.id);
            tag.name = stored != implementation_->tag_names.end() ? std::string_view(stored->second) : tag.name;
            implementation_->allocations.emplace(pointer, implementation::allocation{
                .bytes = bytes,
                .alignment = alignment,
                .domain = domain,
                .tag = tag,
                .world_id = world_id,
                .thread_id = current_thread_id(),
                .stack_id = stack_id
            });
        }
        catch (...)
        {
            // Allocation tracking is diagnostic and must never make a successful allocation fail.
        }
    }
    return { .pointer = pointer };
}

void* memory_system::allocate(
    std::size_t bytes,
    std::size_t alignment,
    memory_domain domain,
    memory_tag tag,
    std::uint64_t world_id)
{
    if (void* pointer = try_allocate(bytes, alignment, domain, tag, world_id))
        return pointer;
    throw std::bad_alloc();
}

void memory_system::deallocate(
    void* pointer,
    std::size_t bytes,
    std::size_t alignment,
    memory_domain domain) noexcept
{
    if (!pointer)
        return;

    if (implementation_->config.track_live_allocations)
    {
        std::lock_guard lock(implementation_->mutex);
        const auto found = implementation_->allocations.find(pointer);
        if (found != implementation_->allocations.end())
        {
            bytes = found->second.bytes;
            alignment = found->second.alignment;
            domain = found->second.domain;
            implementation_->allocations.erase(found);
        }
    }

    implementation_->upstream->deallocate(pointer, bytes, alignment);
    auto& stats = implementation_->stats[domain_index(domain)];
    stats.deallocation_count.fetch_add(1, std::memory_order_relaxed);
    stats.bytes_deallocated.fetch_add(bytes, std::memory_order_relaxed);
    stats.bytes_outstanding.fetch_sub(bytes, std::memory_order_relaxed);
    implementation_->outstanding.fetch_sub(bytes, std::memory_order_relaxed);
    const auto domain_outstanding = stats.bytes_outstanding.load(std::memory_order_relaxed);
    const auto index = domain_index(domain);
    const auto domain_budget = implementation_->budgets[index];
    if (domain_budget.soft_limit == 0 || domain_outstanding < domain_budget.soft_limit)
        implementation_->soft_pressure_announced[index].store(false, std::memory_order_relaxed);
}

void memory_system::set_budget(memory_domain domain, memory_budget budget)
{
    if (domain == memory_domain::count)
        return;
    if (budget.hard_limit != 0 && budget.soft_limit > budget.hard_limit)
        budget.soft_limit = budget.hard_limit;
    std::lock_guard lock(implementation_->mutex);
    implementation_->budgets[domain_index(domain)] = budget;
}

memory_budget memory_system::budget(memory_domain domain) const noexcept
{
    if (domain == memory_domain::count)
        return {};
    std::lock_guard lock(implementation_->mutex);
    return implementation_->budgets[domain_index(domain)];
}

memory_budget memory_system::global_budget() const noexcept
{
    return implementation_->global;
}

std::uint64_t memory_system::add_pressure_handler(memory_pressure_handler handler)
{
    if (!handler)
        return 0;
    std::lock_guard lock(implementation_->mutex);
    const auto token = implementation_->next_handler++;
    implementation_->handlers.emplace(token, std::move(handler));
    return token;
}

bool memory_system::remove_pressure_handler(std::uint64_t token)
{
    std::lock_guard lock(implementation_->mutex);
    return implementation_->handlers.erase(token) != 0;
}

memory_snapshot memory_system::snapshot() const
{
    memory_snapshot result;
    result.sequence = implementation_->snapshot_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    result.physical_memory = implementation_->physical_memory;
    result.global_bytes_outstanding = implementation_->outstanding.load(std::memory_order_relaxed);
    result.global_budget = implementation_->global;
    result.pressure_event_count = implementation_->pressure_events.load(std::memory_order_relaxed);
    result.domains.reserve(domain_count);

    for (std::size_t index = 0; index < domain_count; ++index)
    {
        const auto& source = implementation_->stats[index];
        memory_domain_snapshot domain;
        domain.domain = static_cast<memory_domain>(index);
        domain.stats = {
            .allocation_count = source.allocation_count.load(std::memory_order_relaxed),
            .deallocation_count = source.deallocation_count.load(std::memory_order_relaxed),
            .bytes_allocated = source.bytes_allocated.load(std::memory_order_relaxed),
            .bytes_deallocated = source.bytes_deallocated.load(std::memory_order_relaxed),
            .bytes_outstanding = source.bytes_outstanding.load(std::memory_order_relaxed),
            .peak_bytes_outstanding = source.peak_bytes_outstanding.load(std::memory_order_relaxed)
        };
        domain.budget = implementation_->budgets[index];
        domain.soft_limit_exceeded =
            domain.budget.soft_limit != 0 && domain.stats.bytes_outstanding > domain.budget.soft_limit;
        result.domains.push_back(domain);
    }

    struct aggregate
    {
        memory_tag tag{};
        memory_domain domain{};
        std::size_t count{};
        std::size_t bytes{};
    };
    std::unordered_map<std::uint64_t, aggregate> aggregates;
    using group_key = std::tuple<memory_domain, std::uint32_t, std::uint64_t, std::uint64_t, std::uint64_t>;
    std::map<group_key, memory_allocation_group_snapshot> groups;
    {
        std::lock_guard lock(implementation_->mutex);
        result.tracked_allocation_count = implementation_->allocations.size();
        for (const auto& [_, allocation] : implementation_->allocations)
        {
            const std::uint64_t key =
                (static_cast<std::uint64_t>(allocation.tag.id) << 8u) |
                static_cast<std::uint64_t>(allocation.domain);
            auto& value = aggregates[key];
            value.tag = allocation.tag;
            value.domain = allocation.domain;
            ++value.count;
            value.bytes += allocation.bytes;
            if (allocation.stack_id != 0)
                ++result.sampled_stack_count;

            const group_key group{
                allocation.domain,
                allocation.tag.id,
                allocation.world_id,
                allocation.thread_id,
                allocation.stack_id
            };
            auto& allocation_group = groups[group];
            allocation_group.domain = allocation.domain;
            allocation_group.tag = allocation.tag;
            allocation_group.world_id = allocation.world_id;
            allocation_group.thread_id = allocation.thread_id;
            allocation_group.stack_id = allocation.stack_id;
            ++allocation_group.allocation_count;
            allocation_group.bytes_outstanding += allocation.bytes;
        }
    }
    result.tags.reserve(aggregates.size());
    for (const auto& [_, aggregate] : aggregates)
        result.tags.push_back({
            .tag = aggregate.tag,
            .domain = aggregate.domain,
            .allocation_count = aggregate.count,
            .bytes_outstanding = aggregate.bytes
        });
    std::sort(result.tags.begin(), result.tags.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.bytes_outstanding > rhs.bytes_outstanding;
    });
    result.allocation_groups.reserve(groups.size());
    for (const auto& [_, group] : groups)
        result.allocation_groups.push_back(group);
    std::sort(result.allocation_groups.begin(), result.allocation_groups.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.bytes_outstanding > rhs.bytes_outstanding;
    });
    constexpr std::size_t maximum_snapshot_groups = 256;
    if (result.allocation_groups.size() > maximum_snapshot_groups)
        result.allocation_groups.resize(maximum_snapshot_groups);
    return result;
}

std::vector<memory_leak_record> memory_system::leaks(std::uint64_t world_id) const
{
    std::vector<memory_leak_record> result;
    std::lock_guard lock(implementation_->mutex);
    result.reserve(implementation_->allocations.size());
    for (const auto& [pointer, allocation] : implementation_->allocations)
    {
        if (world_id != 0 && allocation.world_id != world_id)
            continue;
        result.push_back({
            .address = reinterpret_cast<std::uintptr_t>(pointer),
            .bytes = allocation.bytes,
            .alignment = allocation.alignment,
            .domain = allocation.domain,
            .tag = allocation.tag,
            .world_id = allocation.world_id,
            .thread_id = allocation.thread_id,
            .stack_id = allocation.stack_id
        });
    }
    std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.bytes > rhs.bytes;
    });
    return result;
}

const memory_system_config& memory_system::config() const noexcept
{
    return implementation_->config;
}

memory_system& default_memory_system()
{
    static memory_system system;
    return system;
}

system_memory_resource::system_memory_resource(
    memory_system& system,
    memory_domain domain,
    memory_tag tag,
    std::uint64_t world_id) noexcept
    : system_(&system)
    , domain_(domain)
    , tag_(tag)
    , world_id_(world_id)
{
}

memory_domain system_memory_resource::domain() const noexcept
{
    return domain_;
}

std::uint64_t system_memory_resource::world_id() const noexcept
{
    return world_id_;
}

void* system_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment)
{
    return system_->allocate(bytes, alignment, domain_, tag_, world_id_);
}

void system_memory_resource::do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment)
{
    system_->deallocate(pointer, bytes, alignment, domain_);
}

bool system_memory_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept
{
    return this == &other;
}

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

struct linear_arena::block
{
    std::byte* memory{};
    std::size_t capacity{};
    std::size_t offset{};
};

linear_arena::linear_arena(std::size_t initial_capacity, std::pmr::memory_resource* upstream)
    : upstream_(upstream ? upstream : std::pmr::get_default_resource())
    , initial_capacity_(std::max<std::size_t>(initial_capacity, 1024))
{
}

linear_arena::~linear_arena()
{
    release();
}

linear_arena::linear_arena(linear_arena&& other) noexcept
    : blocks_(std::move(other.blocks_))
    , upstream_(other.upstream_)
    , initial_capacity_(other.initial_capacity_)
    , active_block_(other.active_block_)
    , used_(other.used_)
    , peak_used_(other.peak_used_)
    , generation_(other.generation_)
{
    other.blocks_.clear();
    other.used_ = 0;
    other.active_block_ = 0;
}

linear_arena& linear_arena::operator=(linear_arena&& other) noexcept
{
    if (this == &other)
        return *this;
    release();
    blocks_ = std::move(other.blocks_);
    upstream_ = other.upstream_;
    initial_capacity_ = other.initial_capacity_;
    active_block_ = other.active_block_;
    used_ = other.used_;
    peak_used_ = other.peak_used_;
    generation_ = other.generation_;
    other.blocks_.clear();
    other.used_ = 0;
    other.active_block_ = 0;
    return *this;
}

void linear_arena::add_block(std::size_t minimum_capacity)
{
    const std::size_t previous = blocks_.empty() ? initial_capacity_ : blocks_.back().capacity;
    const std::size_t capacity = std::max(minimum_capacity, previous <= std::numeric_limits<std::size_t>::max() / 2
        ? previous * 2
        : minimum_capacity);
    auto* memory = static_cast<std::byte*>(upstream_->allocate(capacity, alignof(std::max_align_t)));
    blocks_.push_back({ .memory = memory, .capacity = capacity, .offset = 0 });
}

void* linear_arena::try_allocate(std::size_t bytes, std::size_t alignment) noexcept
{
    if (bytes == 0)
        bytes = 1;
    if (!valid_alignment(alignment))
        return nullptr;
    try
    {
        for (;;)
        {
            if (blocks_.empty())
                add_block(std::max(initial_capacity_, bytes + alignment));
            auto& value = blocks_[active_block_];
            const auto aligned = (value.offset + alignment - 1) & ~(alignment - 1);
            if (aligned <= value.capacity && bytes <= value.capacity - aligned)
            {
                const auto consumed = aligned + bytes - value.offset;
                value.offset = aligned + bytes;
                used_ += consumed;
                peak_used_ = std::max(peak_used_, used_);
                return value.memory + aligned;
            }
            if (active_block_ + 1 < blocks_.size())
            {
                ++active_block_;
                continue;
            }
            add_block(bytes + alignment);
            ++active_block_;
        }
    }
    catch (...)
    {
        return nullptr;
    }
}

void linear_arena::reset() noexcept
{
    for (auto& value : blocks_)
        value.offset = 0;
    active_block_ = 0;
    used_ = 0;
    ++generation_;
}

arena_mark linear_arena::mark() const noexcept
{
    return {
        .block = active_block_,
        .offset = blocks_.empty() ? 0 : blocks_[active_block_].offset,
        .generation = generation_
    };
}

bool linear_arena::rewind(arena_mark value) noexcept
{
    if (value.generation != generation_ || value.block >= blocks_.size() ||
        value.offset > blocks_[value.block].offset)
        return false;
    for (std::size_t index = value.block + 1; index < blocks_.size(); ++index)
        blocks_[index].offset = 0;
    blocks_[value.block].offset = value.offset;
    active_block_ = value.block;
    used_ = 0;
    for (std::size_t index = 0; index <= active_block_; ++index)
        used_ += blocks_[index].offset;
    return true;
}

std::size_t linear_arena::used() const noexcept
{
    return used_;
}

std::size_t linear_arena::capacity() const noexcept
{
    std::size_t result{};
    for (const auto& value : blocks_)
        result += value.capacity;
    return result;
}

std::size_t linear_arena::peak_used() const noexcept
{
    return peak_used_;
}

std::uint64_t linear_arena::generation() const noexcept
{
    return generation_;
}

void linear_arena::release() noexcept
{
    for (const auto& value : blocks_)
        upstream_->deallocate(value.memory, value.capacity, alignof(std::max_align_t));
    blocks_.clear();
}

void* linear_arena::do_allocate(std::size_t bytes, std::size_t alignment)
{
    if (void* pointer = try_allocate(bytes, alignment))
        return pointer;
    throw std::bad_alloc();
}

void linear_arena::do_deallocate(void*, std::size_t, std::size_t)
{
}

bool linear_arena::do_is_equal(const std::pmr::memory_resource& other) const noexcept
{
    return this == &other;
}

struct fixed_block_pool::implementation
{
    struct free_node
    {
        free_node* next{};
    };
    struct size_class
    {
        std::size_t bytes{};
        free_node* free{};
        std::vector<void*> slabs;
    };

    std::vector<size_class> classes;
    std::size_t blocks_per_slab{};
    std::pmr::memory_resource* upstream{};
    mutable std::mutex mutex;
    std::atomic_size_t pooled{};
    std::atomic_size_t outstanding{};
};

fixed_block_pool::fixed_block_pool(
    std::span<const std::size_t> size_classes,
    std::size_t blocks_per_slab,
    std::pmr::memory_resource* upstream)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->blocks_per_slab = std::max<std::size_t>(blocks_per_slab, 1);
    implementation_->upstream = upstream ? upstream : std::pmr::get_default_resource();
    for (const auto bytes : size_classes)
    {
        if (bytes < sizeof(implementation::free_node))
            continue;
        implementation_->classes.push_back({ .bytes = bytes });
    }
    std::sort(implementation_->classes.begin(), implementation_->classes.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.bytes < rhs.bytes;
    });
}

fixed_block_pool::~fixed_block_pool()
{
    for (auto& size_class : implementation_->classes)
        for (void* slab : size_class.slabs)
            implementation_->upstream->deallocate(
                slab,
                size_class.bytes * implementation_->blocks_per_slab,
                alignof(std::max_align_t));
}

void* fixed_block_pool::try_allocate(std::size_t bytes, std::size_t alignment) noexcept
{
    if (alignment > alignof(std::max_align_t))
        return nullptr;
    std::lock_guard lock(implementation_->mutex);
    const auto found = std::find_if(
        implementation_->classes.begin(),
        implementation_->classes.end(),
        [bytes](const auto& value) { return value.bytes >= bytes; });
    if (found == implementation_->classes.end())
        return nullptr;

    if (!found->free)
    {
        const auto slab_bytes = found->bytes * implementation_->blocks_per_slab;
        void* slab{};
        try
        {
            slab = implementation_->upstream->allocate(slab_bytes, alignof(std::max_align_t));
            found->slabs.push_back(slab);
        }
        catch (...)
        {
            if (slab)
                implementation_->upstream->deallocate(slab, slab_bytes, alignof(std::max_align_t));
            return nullptr;
        }
        auto* cursor = static_cast<std::byte*>(slab);
        for (std::size_t index = 0; index < implementation_->blocks_per_slab; ++index)
        {
            auto* node = reinterpret_cast<implementation::free_node*>(cursor + index * found->bytes);
            node->next = found->free;
            found->free = node;
        }
        implementation_->pooled.fetch_add(slab_bytes, std::memory_order_relaxed);
    }

    auto* result = found->free;
    found->free = result->next;
    implementation_->outstanding.fetch_add(found->bytes, std::memory_order_relaxed);
    return result;
}

void fixed_block_pool::deallocate(void* pointer, std::size_t bytes) noexcept
{
    if (!pointer)
        return;
    std::lock_guard lock(implementation_->mutex);
    const auto found = std::find_if(
        implementation_->classes.begin(),
        implementation_->classes.end(),
        [bytes](const auto& value) { return value.bytes >= bytes; });
    if (found == implementation_->classes.end())
        return;
    auto* node = static_cast<implementation::free_node*>(pointer);
    node->next = found->free;
    found->free = node;
    implementation_->outstanding.fetch_sub(found->bytes, std::memory_order_relaxed);
}

std::size_t fixed_block_pool::pooled_bytes() const noexcept
{
    return implementation_->pooled.load(std::memory_order_relaxed);
}

std::size_t fixed_block_pool::outstanding_bytes() const noexcept
{
    return implementation_->outstanding.load(std::memory_order_relaxed);
}

network_packet_pool::network_packet_pool(std::pmr::memory_resource* upstream)
    : pool_(std::array<std::size_t, 5>{ 256, 512, 1200, 4096, 64 * 1024 }, 64, upstream)
{
}

void* network_packet_pool::try_allocate(std::size_t bytes) noexcept
{
    return pool_.try_allocate(bytes);
}

void network_packet_pool::deallocate(void* pointer, std::size_t bytes) noexcept
{
    pool_.deallocate(pointer, bytes);
}

std::size_t network_packet_pool::outstanding_bytes() const noexcept
{
    return pool_.outstanding_bytes();
}

world_memory_context::world_memory_context(
    memory_system& system,
    std::uint64_t world_id,
    memory_budget budget)
    : system_(&system)
    , world_id_(world_id != 0 ? world_id : allocation_sequence.fetch_add(1, std::memory_order_relaxed) + 1)
    , world_resource_(system, memory_domain::world, make_memory_tag("world"), world_id_)
    , component_resource_(system, memory_domain::components, make_memory_tag("world.components"), world_id_)
{
    if (budget.soft_limit != 0 || budget.hard_limit != 0)
        system.set_budget(memory_domain::world, budget);
}

world_memory_context::~world_memory_context()
{
    const auto outstanding = leaks();
    if (!outstanding.empty())
    {
        std::size_t bytes{};
        for (const auto& leak : outstanding)
            bytes += leak.bytes;
        warn(
            "memory.world",
            "World " + std::to_string(world_id_) + " released with " +
                std::to_string(outstanding.size()) + " tracked allocation(s), " +
                std::to_string(bytes) + " byte(s) outstanding");
    }
}

std::uint64_t world_memory_context::world_id() const noexcept
{
    return world_id_;
}

std::pmr::memory_resource* world_memory_context::world_resource() noexcept
{
    return &world_resource_;
}

std::pmr::memory_resource* world_memory_context::component_resource() noexcept
{
    return &component_resource_;
}

std::vector<memory_leak_record> world_memory_context::leaks() const
{
    return system_->leaks(world_id_);
}

struct streaming_heap::implementation
{
    struct free_block
    {
        std::size_t offset{};
        std::size_t size{};
    };

    implementation(memory_system& memory, std::size_t requested_capacity, memory_tag tag)
        : resource(memory, memory_domain::streaming, tag)
        , capacity(requested_capacity)
    {
        if (capacity != 0)
        {
            storage = static_cast<std::byte*>(resource.allocate(capacity, alignof(std::max_align_t)));
            free.push_back({ .offset = 0, .size = capacity });
        }
    }

    system_memory_resource resource;
    std::byte* storage{};
    std::size_t capacity{};
    std::size_t used{};
    std::size_t peak{};
    std::vector<free_block> free;
    std::unordered_map<void*, std::size_t> allocations;
    mutable std::mutex mutex;
};

streaming_heap::streaming_heap(memory_system& memory, std::size_t capacity, memory_tag tag)
    : implementation_(std::make_unique<implementation>(memory, capacity, tag))
{
}

streaming_heap::~streaming_heap()
{
    if (implementation_->storage)
        implementation_->resource.deallocate(
            implementation_->storage,
            implementation_->capacity,
            alignof(std::max_align_t));
}

void* streaming_heap::try_allocate(std::size_t bytes, std::size_t alignment) noexcept
{
    if (bytes == 0)
        bytes = 1;
    if (!valid_alignment(alignment))
        return nullptr;
    std::lock_guard lock(implementation_->mutex);
    for (auto iterator = implementation_->free.begin(); iterator != implementation_->free.end(); ++iterator)
    {
        const auto aligned = (iterator->offset + alignment - 1) & ~(alignment - 1);
        const auto padding = aligned - iterator->offset;
        if (padding > iterator->size || bytes > iterator->size - padding)
            continue;

        const auto original_end = iterator->offset + iterator->size;
        const auto allocation_end = aligned + bytes;
        const auto original_offset = iterator->offset;
        iterator = implementation_->free.erase(iterator);
        if (padding != 0)
            iterator = implementation_->free.insert(iterator, { .offset = original_offset, .size = padding }) + 1;
        if (allocation_end < original_end)
            implementation_->free.insert(iterator, { .offset = allocation_end, .size = original_end - allocation_end });

        void* pointer = implementation_->storage + aligned;
        implementation_->allocations.emplace(pointer, bytes);
        implementation_->used += bytes;
        implementation_->peak = std::max(implementation_->peak, implementation_->used);
        return pointer;
    }
    return nullptr;
}

std::size_t streaming_heap::capacity() const noexcept
{
    return implementation_->capacity;
}

std::size_t streaming_heap::used() const noexcept
{
    std::lock_guard lock(implementation_->mutex);
    return implementation_->used;
}

std::size_t streaming_heap::peak_used() const noexcept
{
    std::lock_guard lock(implementation_->mutex);
    return implementation_->peak;
}

std::size_t streaming_heap::largest_free_block() const noexcept
{
    std::lock_guard lock(implementation_->mutex);
    std::size_t result{};
    for (const auto& block : implementation_->free)
        result = std::max(result, block.size);
    return result;
}

void* streaming_heap::do_allocate(std::size_t bytes, std::size_t alignment)
{
    if (void* pointer = try_allocate(bytes, alignment))
        return pointer;
    throw std::bad_alloc();
}

void streaming_heap::do_deallocate(void* pointer, std::size_t, std::size_t)
{
    if (!pointer)
        return;
    std::lock_guard lock(implementation_->mutex);
    const auto allocation = implementation_->allocations.find(pointer);
    if (allocation == implementation_->allocations.end())
        return;
    const auto offset = static_cast<std::size_t>(static_cast<std::byte*>(pointer) - implementation_->storage);
    const auto bytes = allocation->second;
    implementation_->allocations.erase(allocation);
    implementation_->used -= bytes;

    auto position = std::lower_bound(
        implementation_->free.begin(),
        implementation_->free.end(),
        offset,
        [](const auto& block, std::size_t value) { return block.offset < value; });
    position = implementation_->free.insert(position, { .offset = offset, .size = bytes });
    if (position != implementation_->free.begin())
    {
        auto previous = position - 1;
        if (previous->offset + previous->size == position->offset)
        {
            previous->size += position->size;
            position = implementation_->free.erase(position);
            position = previous;
        }
    }
    if (position + 1 != implementation_->free.end() &&
        position->offset + position->size == (position + 1)->offset)
    {
        position->size += (position + 1)->size;
        implementation_->free.erase(position + 1);
    }
}

bool streaming_heap::do_is_equal(const std::pmr::memory_resource& other) const noexcept
{
    return this == &other;
}

} // namespace arc
