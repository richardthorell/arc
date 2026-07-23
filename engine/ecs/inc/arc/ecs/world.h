#pragma once

#include <arc/ecs/entity.h>
#include <arc/ecs/reflection.h>
#include <arc/memory/memory.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <span>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::ecs
{

using change_revision = std::uint64_t;

struct change_cursor
{
    change_revision revision{};
};

enum class structural_change_kind : std::uint8_t
{
    entity_created,
    entity_destroyed,
    component_added,
    component_removed
};

struct structural_change
{
    change_revision revision{};
    structural_change_kind kind{};
    entity value{};
    component_type_id component{};
};

struct component_change
{
    entity value{};
    change_revision revision{};
    std::uint64_t fields{};
};

class component_pool_base
{
public:
    virtual ~component_pool_base() = default;
    virtual component_type_id type() const noexcept = 0;
    virtual const component_descriptor& descriptor() const noexcept = 0;
    virtual bool remove(entity value) = 0;
    virtual bool contains(entity value) const noexcept = 0;
    virtual std::span<const entity> entities() const noexcept = 0;
    virtual void ensure_entity_capacity(std::size_t size) = 0;
    virtual void mark(entity value, change_revision revision, std::uint64_t fields) = 0;
    virtual component_change change(entity value) const noexcept = 0;
    virtual std::span<const component_change> change_events() const noexcept = 0;
    virtual std::unique_ptr<component_pool_base> clone(std::pmr::memory_resource* resource) const = 0;
};

template <class T>
class component_pool final : public component_pool_base
{
public:
    explicit component_pool(std::pmr::memory_resource* resource = std::pmr::get_default_resource())
        : resource_(resource ? resource : std::pmr::get_default_resource())
        , sparse_(resource_)
        , entities_(resource_)
        , dense_(resource_)
        , changes_(resource_)
        , change_events_(resource_)
        , pages_(resource_)
        , free_slots_(resource_)
    {
    }

    ~component_pool() override
    {
        for (T* value : dense_)
            std::destroy_at(value);
        for (slot* page : pages_)
            resource_->deallocate(page, sizeof(slot) * page_size, alignof(slot));
    }

    component_pool(const component_pool&) = delete;
    component_pool& operator=(const component_pool&) = delete;

    template <class... Args>
    std::pair<T&, bool> emplace(entity value, change_revision revision, Args&&... args)
    {
        if (contains(value))
        {
            get(value) = T{ std::forward<Args>(args)... };
            mark(value, revision, all_fields);
            return { get(value), false };
        }

        if (free_slots_.empty())
            add_page();
        slot* storage = free_slots_.back();
        free_slots_.pop_back();
        T* component = reinterpret_cast<T*>(storage->storage);
        std::construct_at(component, T{ std::forward<Args>(args)... });
        storage->occupied = true;
        storage->revision = revision;
        storage->fields = all_fields;

        sparse_[value.index] = dense_.size();
        entities_.push_back(value);
        dense_.push_back(component);
        changes_.push_back(storage);
        change_events_.push_back({ value, revision, all_fields });
        return { *component, true };
    }

    T& get(entity value) { return *dense_[sparse_[value.index]]; }
    const T& get(entity value) const { return *dense_[sparse_[value.index]]; }
    T* try_get(entity value) { return contains(value) ? &get(value) : nullptr; }
    const T* try_get(entity value) const { return contains(value) ? &get(value) : nullptr; }

    bool remove(entity value) override
    {
        if (!contains(value))
            return false;

        const std::size_t removed = sparse_[value.index];
        const std::size_t last = dense_.size() - 1;
        T* removed_component = dense_[removed];
        slot* removed_slot = changes_[removed];
        if (removed != last)
        {
            dense_[removed] = dense_[last];
            changes_[removed] = changes_[last];
            entities_[removed] = entities_[last];
            sparse_[entities_[removed].index] = removed;
        }

        dense_.pop_back();
        changes_.pop_back();
        entities_.pop_back();
        sparse_[value.index] = invalid_sparse;
        std::destroy_at(removed_component);
        removed_slot->occupied = false;
        removed_slot->revision = 0;
        removed_slot->fields = 0;
        free_slots_.push_back(removed_slot);
        return true;
    }

    bool contains(entity value) const noexcept override
    {
        return value.valid() &&
            value.index < sparse_.size() &&
            sparse_[value.index] != invalid_sparse &&
            entities_[sparse_[value.index]] == value;
    }

    component_type_id type() const noexcept override { return component_type<T>(); }
    const component_descriptor& descriptor() const noexcept override { return component_metadata<T>(); }

    std::span<const entity> entities() const noexcept override
    {
        return { entities_.data(), entities_.size() };
    }

    void ensure_entity_capacity(std::size_t size) override
    {
        if (sparse_.size() < size)
            sparse_.resize(size, invalid_sparse);
    }

    void mark(entity value, change_revision revision, std::uint64_t fields) override
    {
        if (!contains(value))
            return;
        slot& metadata = *changes_[sparse_[value.index]];
        metadata.revision = revision;
        metadata.fields |= fields;
        change_events_.push_back({ value, revision, fields });
    }

    component_change change(entity value) const noexcept override
    {
        if (!contains(value))
            return {};
        const slot& metadata = *changes_[sparse_[value.index]];
        return { value, metadata.revision, metadata.fields };
    }

    std::span<const component_change> change_events() const noexcept override
    {
        return { change_events_.data(), change_events_.size() };
    }

    std::unique_ptr<component_pool_base> clone(std::pmr::memory_resource* resource) const override
    {
        auto result = std::make_unique<component_pool<T>>(resource);
        result->ensure_entity_capacity(sparse_.size());
        for (std::size_t index = 0; index < dense_.size(); ++index)
        {
            const slot& source = *changes_[index];
            auto [_, inserted] = result->emplace(entities_[index], source.revision, *dense_[index]);
            (void)inserted;
            slot& destination = *result->changes_[index];
            destination.fields = source.fields;
        }
        // A cloned world is a new observation baseline. Retaining the complete
        // source journal makes editor/history snapshots progressively more
        // expensive while conveying no more current-state information than the
        // latest per-instance revision already emitted by emplace above.
        return result;
    }

private:
    static constexpr std::size_t invalid_sparse = static_cast<std::size_t>(-1);
    static constexpr std::size_t page_size = 256;
    static constexpr std::uint64_t all_fields = ~std::uint64_t{};

    struct alignas(alignof(T) > alignof(std::uint64_t) ? alignof(T) : alignof(std::uint64_t)) slot
    {
        alignas(T) std::byte storage[sizeof(T)];
        change_revision revision{};
        std::uint64_t fields{};
        bool occupied{};
    };

    void add_page()
    {
        auto* page = static_cast<slot*>(resource_->allocate(sizeof(slot) * page_size, alignof(slot)));
        pages_.push_back(page);
        for (std::size_t index = 0; index < page_size; ++index)
        {
            std::construct_at(page + index);
            free_slots_.push_back(page + (page_size - index - 1));
        }
    }

    std::pmr::memory_resource* resource_{};
    std::pmr::vector<std::size_t> sparse_;
    std::pmr::vector<entity> entities_;
    std::pmr::vector<T*> dense_;
    std::pmr::vector<slot*> changes_;
    std::pmr::vector<component_change> change_events_;
    std::pmr::vector<slot*> pages_;
    std::pmr::vector<slot*> free_slots_;
};

struct query_signature
{
    std::vector<component_type_id> required;
    std::vector<component_type_id> excluded;

    friend bool operator==(const query_signature&, const query_signature&) = default;
};

template <class T>
struct query_read { using component = T; };
template <class T>
struct query_write { using component = T; };
template <class T>
struct query_optional { using component = T; };
template <class T>
struct query_exclude { using component = T; };

struct query_signature_hash
{
    std::size_t operator()(const query_signature& value) const noexcept
    {
        std::size_t result = 1469598103934665603ull;
        const auto combine = [&result](component_type_id id) {
            result ^= component_type_id_hash{}(id) + 0x9e3779b97f4a7c15ull +
                (result << 6u) + (result >> 2u);
        };
        for (const auto id : value.required)
            combine(id);
        result ^= 0xd6e8feb86659fd93ull;
        for (const auto id : value.excluded)
            combine(id);
        return result;
    }
};

struct query_cache_entry
{
    explicit query_cache_entry(std::pmr::memory_resource* resource)
        : entities(resource)
    {
    }

    query_signature signature;
    std::pmr::vector<entity> entities;
};

class world;

class query_entity_range
{
public:
    class iterator
    {
    public:
        using value_type = entity;
        using difference_type = std::ptrdiff_t;

        entity operator*() const noexcept;
        iterator& operator++() noexcept;
        friend bool operator==(const iterator& lhs, const iterator& rhs) noexcept
        {
            return lhs.index_ == rhs.index_ && lhs.range_ == rhs.range_;
        }

    private:
        iterator(const query_entity_range* range, std::size_t index) noexcept;
        void advance() noexcept;
        const query_entity_range* range_{};
        std::size_t index_{};
        friend class query_entity_range;
    };

    iterator begin() const noexcept { return iterator(this, 0); }
    iterator end() const noexcept { return iterator(this, source_.size()); }
    bool empty() const noexcept { return begin() == end(); }

private:
    query_entity_range(const world& owner, std::span<const entity> source, const query_signature* signature)
        : owner_(&owner), source_(source), signature_(signature)
    {
    }
    const world* owner_{};
    std::span<const entity> source_;
    const query_signature* signature_{};
    friend class world;
    template <class...>
    friend class basic_view;
};

template <class... Components>
class basic_view
{
public:
    query_entity_range entities() const noexcept
    {
        return query_entity_range(*owner_, source_, signature_);
    }

    template <class Function>
    void each(Function&& function) const;

private:
    basic_view(const world& owner, std::span<const entity> source, const query_signature* signature)
        : owner_(&owner), source_(source), signature_(signature)
    {
    }
    const world* owner_{};
    std::span<const entity> source_;
    const query_signature* signature_{};
    friend class world;
};

template <class T>
class component_change_range
{
public:
    class iterator
    {
    public:
        component_change operator*() const noexcept { return events_[index_]; }
        iterator& operator++() noexcept { ++index_; advance(); return *this; }
        friend bool operator==(const iterator& lhs, const iterator& rhs) noexcept
        {
            return lhs.events_.data() == rhs.events_.data() && lhs.index_ == rhs.index_;
        }

    private:
        iterator(std::span<const component_change> events, change_revision since, std::size_t index)
            : events_(events), since_(since), index_(index) { advance(); }
        void advance() noexcept
        {
            while (index_ < events_.size() && events_[index_].revision <= since_)
                ++index_;
        }
        std::span<const component_change> events_;
        change_revision since_{};
        std::size_t index_{};
        friend class component_change_range;
    };

    iterator begin() const noexcept { return iterator(events_, since_, 0); }
    iterator end() const noexcept { return iterator(events_, since_, events_.size()); }

private:
    component_change_range(const component_pool<T>* pool, change_revision since)
        : events_(pool ? pool->change_events() : std::span<const component_change>{}), since_(since) {}
    std::span<const component_change> events_;
    change_revision since_{};
    friend class world;
};

/** Stable-address sparse-set ECS world with prepared allocation-free queries. */
class world
{
public:
    world()
        : memory_(std::make_shared<world_memory_context>())
    {
    }

    explicit world(memory_system& memory, std::uint64_t world_id = 0, memory_budget budget = {})
        : memory_(std::make_shared<world_memory_context>(memory, world_id, budget))
    {
    }

    world(const world& other)
        : memory_(other.memory_)
        , generations_(other.generations_)
        , alive_(other.alive_)
        , free_list_(other.free_list_)
        , structural_changes_(other.structural_changes_)
        , live_count_(other.live_count_)
        , revision_(other.revision_)
    {
        for (const auto& [key, values] : other.pools_)
            pools_.emplace(key, values->clone(memory_->component_resource()));
        for (const auto& query : other.query_cache_)
            prepare_query(query->signature);
    }

    world& operator=(const world& other)
    {
        if (this == &other)
            return *this;
        world copy(other);
        swap(copy);
        return *this;
    }

    world(world&&) noexcept = default;
    world& operator=(world&& other) noexcept
    {
        if (this == &other)
            return *this;
        world replacement(std::move(other));
        swap(replacement);
        return *this;
    }

    void swap(world& other) noexcept
    {
        generations_.swap(other.generations_);
        alive_.swap(other.alive_);
        free_list_.swap(other.free_list_);
        pools_.swap(other.pools_);
        query_cache_.swap(other.query_cache_);
        structural_changes_.swap(other.structural_changes_);
        memory_.swap(other.memory_);
        std::swap(live_count_, other.live_count_);
        std::swap(revision_, other.revision_);
        std::swap(structural_lock_depth_, other.structural_lock_depth_);
        std::swap(flushing_commands_, other.flushing_commands_);
    }

    entity create()
    {
        assert_structural_mutation_allowed();
        std::uint32_t index{};
        if (!free_list_.empty())
        {
            index = free_list_.back();
            free_list_.pop_back();
        }
        else
        {
            index = static_cast<std::uint32_t>(generations_.size());
            generations_.push_back(1);
            alive_.push_back(false);
            for (auto& [_, pool] : pools_)
                pool->ensure_entity_capacity(generations_.size());
        }
        alive_[index] = true;
        ++live_count_;
        const entity result{ index, generations_[index] };
        record_structural(structural_change_kind::entity_created, result, {});
        refresh_queries(result);
        return result;
    }

    bool destroy(entity value)
    {
        assert_structural_mutation_allowed();
        if (!alive(value))
            return false;
        for (auto& [type, pool] : pools_)
            if (pool->remove(value))
                record_structural(structural_change_kind::component_removed, value, type);
        remove_from_queries(value);
        alive_[value.index] = false;
        ++generations_[value.index];
        free_list_.push_back(value.index);
        --live_count_;
        record_structural(structural_change_kind::entity_destroyed, value, {});
        return true;
    }

    bool alive(entity value) const noexcept
    {
        return value.valid() && value.index < generations_.size() && alive_[value.index] &&
            generations_[value.index] == value.generation;
    }

    std::size_t live_count() const noexcept { return live_count_; }
    change_revision revision() const noexcept
    {
        return std::atomic_ref<const change_revision>(revision_).load(std::memory_order_acquire);
    }
    world_memory_context& memory() noexcept { return *memory_; }
    const world_memory_context& memory() const noexcept { return *memory_; }

    std::vector<entity> entities_snapshot() const
    {
        std::vector<entity> result;
        result.reserve(live_count_);
        for (std::uint32_t index = 0; index < generations_.size(); ++index)
            if (alive_[index])
                result.push_back({ index, generations_[index] });
        return result;
    }

    /** Compatibility snapshot; hot paths should use prepared queries. */
    std::vector<entity> entities() const { return entities_snapshot(); }

    template <class T, class... Args>
    T& emplace(entity value, Args&&... args)
    {
        assert_structural_mutation_allowed();
        if (!alive(value))
            throw std::invalid_argument("cannot add a component to a stale entity");
        const change_revision change = next_revision();
        auto [component, inserted] = pool<T>().emplace(value, change, std::forward<Args>(args)...);
        if (inserted)
            record_structural_at(change, structural_change_kind::component_added, value, component_type<T>());
        refresh_queries(value);
        return component;
    }

    template <class T>
    T& get(entity value)
    {
        T* component = try_get_untracked<T>(value);
        if (!component)
            throw std::out_of_range("component does not exist for entity");
        mark_dirty<T>(value);
        return *component;
    }

    template <class T>
    const T& get(entity value) const
    {
        const T* component = try_get<T>(value);
        if (!component)
            throw std::out_of_range("component does not exist for entity");
        return *component;
    }

    template <class T>
    T* try_get(entity value)
    {
        T* component = try_get_untracked<T>(value);
        if (component)
            mark_dirty<T>(value);
        return component;
    }

    template <class T>
    const T* try_get(entity value) const
    {
        const auto* values = try_pool<T>();
        return values ? values->try_get(value) : nullptr;
    }

    template <class T>
    bool has(entity value) const { return has(value, component_type<T>()); }

    bool has(entity value, component_type_id type) const noexcept
    {
        const auto found = pools_.find(type);
        return found != pools_.end() && found->second->contains(value);
    }

    template <class T>
    bool remove(entity value)
    {
        assert_structural_mutation_allowed();
        return remove(value, component_type<T>());
    }

    bool remove(entity value, component_type_id type)
    {
        assert_structural_mutation_allowed();
        const auto found = pools_.find(type);
        if (found == pools_.end() || !found->second->remove(value))
            return false;
        record_structural(structural_change_kind::component_removed, value, type);
        refresh_queries(value);
        return true;
    }

    template <class T>
    void mark_dirty(entity value, std::uint64_t fields = ~std::uint64_t{})
    {
        if (auto* values = try_pool<T>())
            values->mark(value, next_revision(), fields);
    }

    template <class T, class Function>
    bool patch_field(entity value, component_field_id field, Function&& function)
    {
        T* component = try_get_untracked<T>(value);
        if (!component)
            return false;
        const std::size_t index = field_index(component_metadata<T>(), field);
        if (index == static_cast<std::size_t>(-1))
            return false;
        std::forward<Function>(function)(*component);
        const std::uint64_t mask = index < 64 ? (std::uint64_t{ 1 } << index) : ~std::uint64_t{};
        pool<T>().mark(value, next_revision(), mask);
        return true;
    }

    template <class T>
    component_change_range<T> changes_since(change_cursor cursor) const noexcept
    {
        return component_change_range<T>(try_pool<T>(), cursor.revision);
    }

    std::span<const structural_change> structural_changes() const noexcept
    {
        return { structural_changes_.data(), structural_changes_.size() };
    }

    template <class... Components>
    void prepare_query()
    {
        query_signature signature = typed_signature<Components...>();
        if (find_query(signature))
            return;
        auto entry = std::make_unique<query_cache_entry>(memory_->component_resource());
        entry->signature = std::move(signature);
        entry->entities.reserve(live_count_);
        for (std::uint32_t index = 0; index < generations_.size(); ++index)
        {
            const entity value{ index, generations_[index] };
            if (alive(value) && matches(value, entry->signature))
                entry->entities.push_back(value);
        }
        query_cache_.emplace_back(std::move(entry));
    }

    void prepare_query(query_signature signature)
    {
        canonicalize(signature);
        if (find_query(signature))
            return;
        auto entry = std::make_unique<query_cache_entry>(memory_->component_resource());
        entry->signature = std::move(signature);
        entry->entities.reserve(live_count_);
        for (std::uint32_t index = 0; index < generations_.size(); ++index)
        {
            const entity value{ index, generations_[index] };
            if (alive(value) && matches(value, entry->signature))
                entry->entities.push_back(value);
        }
        query_cache_.emplace_back(std::move(entry));
    }

    query_entity_range query(const query_signature& signature) const
    {
        const auto* cached = find_query(signature);
        return { *this, cached ? std::span<const entity>(cached->entities) : std::span<const entity>{},
            cached ? &cached->signature : nullptr };
    }

    template <class... Specifications>
    void prepare_typed_query()
    {
        prepare_query(access_signature<Specifications...>());
    }

    template <class... Specifications>
    query_entity_range query() const
    {
        const query_signature& signature = access_signature<Specifications...>();
        if (const auto* cached = find_query(signature))
            return { *this, cached->entities, &cached->signature };
        return { *this, {}, nullptr };
    }

    template <class... Components>
    basic_view<Components...> view() const
    {
        static_assert(sizeof...(Components) > 0);
        const query_signature& temporary = typed_signature<Components...>();
        if (const auto* cached = find_query(temporary))
            return { *this, cached->entities, &cached->signature };

        const component_pool_base* driver = nullptr;
        std::size_t smallest = static_cast<std::size_t>(-1);
        for (const component_type_id type : temporary.required)
        {
            const auto found = pools_.find(type);
            if (found == pools_.end())
                return { *this, {}, nullptr };
            const auto size = found->second->entities().size();
            if (size < smallest)
            {
                driver = found->second.get();
                smallest = size;
            }
        }
        return { *this, driver ? driver->entities() : std::span<const entity>{}, &temporary };
    }

    bool matches(entity value, const query_signature& signature) const noexcept
    {
        if (!alive(value))
            return false;
        for (const auto type : signature.required)
            if (!has(value, type))
                return false;
        for (const auto type : signature.excluded)
            if (has(value, type))
                return false;
        return true;
    }

    void begin_scheduled_execution() noexcept { ++structural_lock_depth_; }
    void end_scheduled_execution() noexcept { if (structural_lock_depth_) --structural_lock_depth_; }
    bool structural_mutation_locked() const noexcept { return structural_lock_depth_ != 0; }

private:
    template <class T>
    component_pool<T>& pool()
    {
        const component_type_id key = component_type<T>();
        auto found = pools_.find(key);
        if (found == pools_.end())
        {
            auto values = std::make_unique<component_pool<T>>(memory_->component_resource());
            values->ensure_entity_capacity(generations_.size());
            found = pools_.emplace(key, std::move(values)).first;
        }
        auto& values = static_cast<component_pool<T>&>(*found->second);
        values.ensure_entity_capacity(generations_.size());
        return values;
    }

    template <class T>
    component_pool<T>* try_pool()
    {
        const auto found = pools_.find(component_type<T>());
        return found == pools_.end() ? nullptr : static_cast<component_pool<T>*>(found->second.get());
    }

    template <class T>
    const component_pool<T>* try_pool() const
    {
        const auto found = pools_.find(component_type<T>());
        return found == pools_.end() ? nullptr : static_cast<const component_pool<T>*>(found->second.get());
    }

    template <class T>
    T* try_get_untracked(entity value)
    {
        auto* values = try_pool<T>();
        return values ? values->try_get(value) : nullptr;
    }

    template <class... Components>
    static const query_signature& typed_signature()
    {
        static const query_signature result = [] {
            query_signature value;
            value.required = { component_type<Components>()... };
            std::sort(value.required.begin(), value.required.end());
            value.required.erase(std::unique(value.required.begin(), value.required.end()), value.required.end());
            return value;
        }();
        return result;
    }

    template <class Specification>
    static void append_access(query_signature& signature)
    {
        using component = typename Specification::component;
        if constexpr (std::is_same_v<Specification, query_exclude<component>>)
            signature.excluded.push_back(component_type<component>());
        else if constexpr (!std::is_same_v<Specification, query_optional<component>>)
            signature.required.push_back(component_type<component>());
    }

    template <class... Specifications>
    static const query_signature& access_signature()
    {
        static const query_signature result = [] {
            query_signature value;
            (append_access<Specifications>(value), ...);
            canonicalize(value);
            return value;
        }();
        return result;
    }

    static void canonicalize(query_signature& signature)
    {
        const auto sort_unique = [](auto& values) {
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        };
        sort_unique(signature.required);
        sort_unique(signature.excluded);
    }

    query_cache_entry* find_query(const query_signature& signature) noexcept
    {
        for (auto& entry : query_cache_)
            if (entry->signature == signature)
                return entry.get();
        return nullptr;
    }

    const query_cache_entry* find_query(const query_signature& signature) const noexcept
    {
        for (const auto& entry : query_cache_)
            if (entry->signature == signature)
                return entry.get();
        return nullptr;
    }

    void refresh_queries(entity value)
    {
        for (auto& entry : query_cache_)
        {
            const bool should_contain = matches(value, entry->signature);
            const auto found = std::lower_bound(entry->entities.begin(), entry->entities.end(), value,
                [](entity lhs, entity rhs) { return lhs.index < rhs.index; });
            const bool contains = found != entry->entities.end() && *found == value;
            if (should_contain && !contains)
                entry->entities.insert(found, value);
            else if (!should_contain && contains)
                entry->entities.erase(found);
        }
    }

    void remove_from_queries(entity value)
    {
        for (auto& entry : query_cache_)
        {
            const auto found = std::lower_bound(entry->entities.begin(), entry->entities.end(), value,
                [](entity lhs, entity rhs) { return lhs.index < rhs.index; });
            if (found != entry->entities.end() && *found == value)
                entry->entities.erase(found);
        }
    }

    change_revision next_revision() noexcept
    {
        auto atomic_revision = std::atomic_ref<change_revision>(revision_);
        change_revision result = atomic_revision.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (result == 0)
            result = atomic_revision.fetch_add(1, std::memory_order_acq_rel) + 1;
        return result;
    }

    void record_structural(structural_change_kind kind, entity value, component_type_id component)
    {
        record_structural_at(next_revision(), kind, value, component);
    }

    void record_structural_at(
        change_revision revision,
        structural_change_kind kind,
        entity value,
        component_type_id component)
    {
        structural_changes_.push_back({ revision, kind, value, component });
    }

    void assert_structural_mutation_allowed() const
    {
        if (structural_lock_depth_ != 0 && !flushing_commands_)
            throw std::logic_error("direct structural ECS mutation is forbidden during scheduled execution");
    }

    void begin_command_flush() noexcept { flushing_commands_ = true; }
    void end_command_flush() noexcept { flushing_commands_ = false; }

    std::shared_ptr<world_memory_context> memory_;
    std::vector<std::uint32_t> generations_;
    std::vector<bool> alive_;
    std::vector<std::uint32_t> free_list_;
    std::unordered_map<component_type_id, std::unique_ptr<component_pool_base>, component_type_id_hash> pools_;
    std::vector<std::unique_ptr<query_cache_entry>> query_cache_;
    std::vector<structural_change> structural_changes_;
    std::size_t live_count_{};
    change_revision revision_{};
    std::uint32_t structural_lock_depth_{};
    bool flushing_commands_{};
    friend class query_entity_range::iterator;
    friend class entity_command_buffer;
};

inline query_entity_range::iterator::iterator(const query_entity_range* range, std::size_t index) noexcept
    : range_(range), index_(index)
{
    advance();
}

inline void query_entity_range::iterator::advance() noexcept
{
    if (!range_ || !range_->signature_)
        return;
    while (index_ < range_->source_.size() &&
        !range_->owner_->matches(range_->source_[index_], *range_->signature_))
        ++index_;
}

inline entity query_entity_range::iterator::operator*() const noexcept
{
    return range_->source_[index_];
}

inline query_entity_range::iterator& query_entity_range::iterator::operator++() noexcept
{
    ++index_;
    advance();
    return *this;
}

template <class... Components>
template <class Function>
void basic_view<Components...>::each(Function&& function) const
{
    for (const entity value : entities())
        std::forward<Function>(function)(value, std::as_const(*owner_).template get<Components>(value)...);
}

} // namespace arc::ecs
