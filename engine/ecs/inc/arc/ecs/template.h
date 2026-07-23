#pragma once

#include <arc/ecs/command_buffer.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace arc::ecs
{

class entity_template
{
public:
    struct component_factory
    {
        component_type_id type{};
        std::function<void(entity_command_buffer&, entity_target)> construct;
    };

    std::string name;
    std::vector<component_factory> components;
    std::vector<entity_template> children;

    deferred_entity instantiate(entity_command_buffer& commands, entity_target parent = entity{}) const
    {
        const deferred_entity value = commands.create();
        for (const component_factory& component : components)
            component.construct(commands, value);
        for (const entity_template& child : children)
            child.instantiate(commands, value);
        (void)parent; // Relationship binding is supplied by the scene hierarchy adapter.
        return value;
    }
};

class entity_template_builder
{
public:
    explicit entity_template_builder(std::string name = {})
    {
        value_.name = std::move(name);
    }

    template <class T>
    entity_template_builder& component(T value = {})
    {
        value_.components.push_back({
            component_type<T>(),
            [value = std::move(value)](entity_command_buffer& commands, entity_target target) mutable {
                commands.add<T>(target, value);
            }
        });
        return *this;
    }

    entity_template_builder& child(entity_template child)
    {
        value_.children.push_back(std::move(child));
        return *this;
    }

    entity_template build() { return std::move(value_); }

private:
    entity_template value_;
};

} // namespace arc::ecs
