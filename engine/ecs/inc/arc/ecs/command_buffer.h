#pragma once

#include <arc/ecs/world.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::ecs
{

struct deferred_entity
{
    std::uint64_t buffer{};
    std::uint32_t ordinal{};

    constexpr bool valid() const noexcept { return buffer != 0; }
    friend constexpr bool operator==(deferred_entity, deferred_entity) noexcept = default;
};

class entity_target
{
public:
    entity_target(entity value) : immediate_(value) {}
    entity_target(deferred_entity value) : deferred_(value), is_deferred_(true) {}

    entity resolve(const std::unordered_map<std::uint64_t, std::vector<entity>>& values) const noexcept
    {
        if (!is_deferred_)
            return immediate_;
        const auto found = values.find(deferred_.buffer);
        if (found == values.end() || deferred_.ordinal >= found->second.size())
            return {};
        return found->second[deferred_.ordinal];
    }

private:
    entity immediate_{};
    deferred_entity deferred_{};
    bool is_deferred_{};
};

enum class command_error_code : std::uint8_t
{
    unresolved_entity,
    stale_entity,
    component_missing,
    exception
};

struct command_error
{
    std::size_t command_index{};
    command_error_code code{};
    std::string message;
};

struct command_flush_result
{
    std::size_t applied{};
    std::vector<command_error> errors;

    bool succeeded() const noexcept { return errors.empty(); }
};

/**
 * Thread-local structural mutation recorder. Buffers are sorted by their
 * deterministic key before a phase flush.
 */
class entity_command_buffer
{
public:
    struct sort_key
    {
        std::uint32_t phase{};
        std::uint32_t system{};
        std::uint32_t partition{};

        friend constexpr auto operator<=>(sort_key, sort_key) noexcept = default;
    };

    explicit entity_command_buffer(sort_key key = {})
        : key_(key)
        , id_(next_id())
    {
    }

    entity_command_buffer(const entity_command_buffer&) = delete;
    entity_command_buffer& operator=(const entity_command_buffer&) = delete;
    entity_command_buffer(entity_command_buffer&&) noexcept = default;
    entity_command_buffer& operator=(entity_command_buffer&&) noexcept = default;

    deferred_entity create()
    {
        const deferred_entity result{ id_, created_++ };
        commands_.push_back({
            [result](world& owner, resolver& values) {
                auto& buffer_values = values[result.buffer];
                if (buffer_values.size() <= result.ordinal)
                    buffer_values.resize(result.ordinal + 1);
                buffer_values[result.ordinal] = owner.create();
                return true;
            },
            "create entity"
        });
        return result;
    }

    void destroy(entity_target target)
    {
        record_target(target, "destroy entity", [](world& owner, entity value) {
            return owner.destroy(value);
        });
    }

    template <class T>
    void add(entity_target target, T component)
    {
        record_target(target, "add component", [component = std::move(component)](world& owner, entity value) mutable {
            owner.emplace<T>(value, std::move(component));
            return true;
        });
    }

    template <class T>
    void remove(entity_target target)
    {
        record_target(target, "remove component", [](world& owner, entity value) {
            return owner.remove<T>(value);
        });
    }

    template <class T, class Function>
    void patch(entity_target target, Function&& function)
    {
        record_target(target, "patch component",
            [function = std::forward<Function>(function)](world& owner, entity value) mutable {
                T* component = owner.template try_get<T>(value);
                if (!component)
                    return false;
                std::invoke(function, *component);
                owner.template mark_dirty<T>(value);
                return true;
            });
    }

    bool empty() const noexcept { return commands_.empty(); }
    std::size_t size() const noexcept { return commands_.size(); }
    sort_key key() const noexcept { return key_; }
    std::uint64_t id() const noexcept { return id_; }

    command_flush_result flush(world& owner)
    {
        std::vector<entity_command_buffer*> buffers{ this };
        return flush_ordered(owner, buffers);
    }

    static command_flush_result flush_ordered(world& owner, std::span<entity_command_buffer*> buffers)
    {
        std::vector<entity_command_buffer*> ordered(buffers.begin(), buffers.end());
        std::stable_sort(ordered.begin(), ordered.end(), [](const auto* lhs, const auto* rhs) {
            return lhs->key_ < rhs->key_;
        });

        command_flush_result result;
        resolver resolved;
        owner.begin_command_flush();
        std::size_t command_index{};
        for (entity_command_buffer* buffer : ordered)
        {
            for (command& value : buffer->commands_)
            {
                try
                {
                    if (value.apply(owner, resolved))
                        ++result.applied;
                    else
                        result.errors.push_back({
                            command_index, command_error_code::stale_entity,
                            value.label + " targeted a stale entity or missing component" });
                }
                catch (const std::exception& exception)
                {
                    result.errors.push_back({
                        command_index, command_error_code::exception,
                        value.label + ": " + exception.what() });
                }
                ++command_index;
            }
            buffer->commands_.clear();
            buffer->created_ = 0;
        }
        owner.end_command_flush();
        return result;
    }

private:
    using resolver = std::unordered_map<std::uint64_t, std::vector<entity>>;

    struct command
    {
        std::function<bool(world&, resolver&)> apply;
        std::string label;
    };

    template <class Function>
    void record_target(entity_target target, std::string label, Function&& function)
    {
        commands_.push_back({
            [target, function = std::forward<Function>(function)](world& owner, resolver& values) mutable {
                const entity resolved = target.resolve(values);
                return resolved.valid() && owner.alive(resolved) && std::invoke(function, owner, resolved);
            },
            std::move(label)
        });
    }

    static std::uint64_t next_id() noexcept
    {
        static std::atomic<std::uint64_t> value{ 1 };
        return value.fetch_add(1, std::memory_order_relaxed);
    }

    sort_key key_{};
    std::uint64_t id_{};
    std::uint32_t created_{};
    std::vector<command> commands_;
};

} // namespace arc::ecs
