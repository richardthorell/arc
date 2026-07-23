#pragma once

#include <arc/ecs/world.h>

namespace arc::scene
{

using registry = ecs::world;
using component_pool_base = ecs::component_pool_base;

template <class T>
using component_pool = ecs::component_pool<T>;

template <class... Components>
using basic_view = ecs::basic_view<Components...>;

} // namespace arc::scene
