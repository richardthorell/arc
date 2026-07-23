#pragma once

#include <arc/ecs/identity.h>

namespace arc::scene
{

using entity_guid = ecs::entity_guid;
using entity_guid_hash = ecs::entity_guid_hash;
using ecs::generate_entity_guid;
using ecs::parse_entity_guid;
using ecs::to_string;

} // namespace arc::scene
