#pragma once

#include <arc/scene/entity.h>
#include <arc/memory/memory.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::scene
{

class component_pool_base
{
public:
    virtual ~component_pool_base() = default;
    virtual void remove(entity value) = 0;
    virtual bool contains(entity value) const = 0;
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
    T& emplace(entity value, Args&&... args)
    {
        if (contains(value))
            return get(value) = T{ std::forward<Args>(args)... };

        if (free_slots_.empty())
            add_page();
        slot* storage = free_slots_.back();
        free_slots_.pop_back();
        T* component = reinterpret_cast<T*>(storage->storage);
        std::construct_at(component, T{ std::forward<Args>(args)... });
        storage->occupied = true;

        sparse_[value.index] = dense_.size();
        entities_.push_back(value);
        dense_.push_back(component);
        return *component;
    }

    T& get(entity value)
    {
        return *dense_[sparse_[value.index]];
    }

    const T& get(entity value) const
    {
        return *dense_[sparse_[value.index]];
    }

    T* try_get(entity value)
    {
        return contains(value) ? &get(value) : nullptr;
    }

    const T* try_get(entity value) const
    {
        return contains(value) ? &get(value) : nullptr;
    }

    void remove(entity value) override
    {
        if (!contains(value))
            return;

        const std::size_t removed = sparse_[value.index];
        const std::size_t last = dense_.size() - 1;
        T* removed_component = dense_[removed];
        if (removed != last)
        {
            dense_[removed] = dense_[last];
            entities_[removed] = entities_[last];
            sparse_[entities_[removed].index] = removed;
        }

        dense_.pop_back();
        entities_.pop_back();
        sparse_[value.index] = invalid_sparse;
        std::destroy_at(removed_component);
        auto* storage = reinterpret_cast<slot*>(removed_component);
        storage->occupied = false;
        free_slots_.push_back(storage);
    }

    bool contains(entity value) const override
    {
        return value.valid() &&
            value.index < sparse_.size() &&
            sparse_[value.index] != invalid_sparse &&
            entities_[sparse_[value.index]] == value;
    }

    std::unique_ptr<component_pool_base> clone(std::pmr::memory_resource* resource) const override
    {
        auto result = std::make_unique<component_pool<T>>(resource);
        result->ensure_entity_capacity(sparse_.size());
        for (std::size_t index = 0; index < dense_.size(); ++index)
            result->emplace(entities_[index], *dense_[index]);
        return result;
    }

    const std::pmr::vector<entity>& entities() const noexcept
    {
        return entities_;
    }

    void ensure_entity_capacity(std::size_t size)
    {
        if (sparse_.size() < size)
            sparse_.resize(size, invalid_sparse);
    }

private:
    static constexpr std::size_t invalid_sparse = static_cast<std::size_t>(-1);
    static constexpr std::size_t page_size = 256;

    struct alignas(T) slot
    {
        alignas(T) std::byte storage[sizeof(T)];
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
    std::pmr::vector<slot*> pages_;
    std::pmr::vector<slot*> free_slots_;
};

class registry;

template <class... Components>
class basic_view
{
public:
    basic_view(const registry& owner, std::vector<entity> entities)
        : owner_(&owner)
        , entities_(std::move(entities))
    {
    }

    const std::vector<entity>& entities() const noexcept
    {
        return entities_;
    }

    template <class Func>
    void each(Func&& func) const;

private:
    const registry* owner_{};
    std::vector<entity> entities_;
};

/**
 * @brief Minimal sparse-set ECS registry.
 */
class registry
{
public:
    registry()
        : memory_(std::make_shared<world_memory_context>())
    {
    }

    explicit registry(memory_system& memory, std::uint64_t world_id = 0, memory_budget budget = {})
        : memory_(std::make_shared<world_memory_context>(memory, world_id, budget))
    {
    }

    registry(const registry& other)
        : memory_(other.memory_)
        , generations_(other.generations_)
        , alive_(other.alive_)
        , free_list_(other.free_list_)
        , live_count_(other.live_count_)
    {
        for (const auto& [key, values] : other.pools_)
            pools_.emplace(key, values->clone(memory_->component_resource()));
    }

    registry& operator=(const registry& other)
    {
        if (this == &other)
            return *this;
        registry copy(other);
        swap(copy);
        return *this;
    }

    registry(registry&&) noexcept = default;
    registry& operator=(registry&&) noexcept = default;

    void swap(registry& other) noexcept
    {
        generations_.swap(other.generations_);
        alive_.swap(other.alive_);
        free_list_.swap(other.free_list_);
        pools_.swap(other.pools_);
        memory_.swap(other.memory_);
        std::swap(live_count_, other.live_count_);
    }

    /**
     * @brief Create a new live entity.
     */
    entity create()
    {
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
        }

        alive_[index] = true;
        ++live_count_;
        return { .index = index, .generation = generations_[index] };
    }

    /**
     * @brief Destroy a live entity and remove all of its components.
     */
    bool destroy(entity value)
    {
        if (!alive(value))
            return false;

        for (auto& [_, pool] : pools_)
            pool->remove(value);

        alive_[value.index] = false;
        ++generations_[value.index];
        free_list_.push_back(value.index);
        --live_count_;
        return true;
    }

    /**
     * @brief Return whether an entity is currently live.
     */
    bool alive(entity value) const noexcept
    {
        return value.valid() &&
            value.index < generations_.size() &&
            alive_[value.index] &&
            generations_[value.index] == value.generation;
    }

    /**
     * @brief Return the number of live entities.
     */
    std::size_t live_count() const noexcept
    {
        return live_count_;
    }

    world_memory_context& memory() noexcept
    {
        return *memory_;
    }

    const world_memory_context& memory() const noexcept
    {
        return *memory_;
    }

    /** Return a stable snapshot of all currently live entity handles. */
    std::vector<entity> entities() const
    {
        std::vector<entity> result;
        result.reserve(live_count_);
        for (std::uint32_t index = 0; index < generations_.size(); ++index)
            if (alive_[index])
                result.push_back({ index, generations_[index] });
        return result;
    }

    template <class T, class... Args>
    T& emplace(entity value, Args&&... args)
    {
        if (!alive(value))
            throw std::invalid_argument("cannot add a component to a stale entity");

        return pool<T>().emplace(value, std::forward<Args>(args)...);
    }

    template <class T>
    T& get(entity value)
    {
        T* component = try_get<T>(value);
        if (!component)
            throw std::out_of_range("component does not exist for entity");
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
        auto* values = try_pool<T>();
        return values ? values->try_get(value) : nullptr;
    }

    template <class T>
    const T* try_get(entity value) const
    {
        const auto* values = try_pool<T>();
        return values ? values->try_get(value) : nullptr;
    }

    template <class T>
    bool has(entity value) const
    {
        const auto* values = try_pool<T>();
        return values && values->contains(value);
    }

    template <class T>
    void remove(entity value)
    {
        if (auto* values = try_pool<T>())
            values->remove(value);
    }

    template <class... Components>
    basic_view<Components...> view() const
    {
        std::vector<entity> result;
        const auto* first = try_pool<std::tuple_element_t<0, std::tuple<Components...>>>();
        if (!first)
            return { *this, result };

        for (const entity value : first->entities())
        {
            if ((has<Components>(value) && ...))
                result.push_back(value);
        }
        return { *this, std::move(result) };
    }

private:
    template <class T>
    component_pool<T>& pool()
    {
        const std::type_index key(typeid(T));
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
        const std::type_index key(typeid(T));
        auto found = pools_.find(key);
        if (found == pools_.end())
            return nullptr;
        return static_cast<component_pool<T>*>(found->second.get());
    }

    template <class T>
    const component_pool<T>* try_pool() const
    {
        const std::type_index key(typeid(T));
        auto found = pools_.find(key);
        if (found == pools_.end())
            return nullptr;
        return static_cast<const component_pool<T>*>(found->second.get());
    }

    std::shared_ptr<world_memory_context> memory_;
    std::vector<std::uint32_t> generations_;
    std::vector<bool> alive_;
    std::vector<std::uint32_t> free_list_;
    std::unordered_map<std::type_index, std::unique_ptr<component_pool_base>> pools_;
    std::size_t live_count_{};
};

template <class... Components>
template <class Func>
void basic_view<Components...>::each(Func&& func) const
{
    for (const entity value : entities_)
        std::forward<Func>(func)(value, owner_->template get<Components>(value)...);
}

} // namespace arc::scene
