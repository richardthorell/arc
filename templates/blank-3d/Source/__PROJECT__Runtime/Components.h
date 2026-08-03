#pragma once

#include <arc/project/reflection.h>

namespace {{PROJECT_TOKEN}}
{

ARC_COMPONENT("91d8467f81dc4f26a40f065463e82941", 1, "Gameplay Stats", "Gameplay",
              "Example project-owned component generated into ARC reflection metadata.")
struct gameplay_stats_component
{
    ARC_PROPERTY("6b3ad5fa49db01a1", "Health", "Vitals", "Current entity health.", "float", "100.0", "0", "1000",
                 "editable|serialized|save_game|prefab", "", "")
    float health{100.0F};

    ARC_PROPERTY("1af2f81066c4fd42", "Invulnerable", "Vitals", "Whether damage is currently ignored.", "bool", "false", "", "",
                 "editable|serialized|prefab", "", "")
    bool invulnerable{};
};

} // namespace {{PROJECT_TOKEN}}
