#pragma once

#include <arc/project/reflection.h>

namespace arc::project::tests
{

ARC_COMPONENT("cb3208e9cd18443693a80cbed1099ccd", 3, "Test Stats", "Tests",
              "Reflected fixture used to verify project code generation.")
struct reflected_stats
{
    ARC_PROPERTY("2b880a80f9e8fd40", "Count", "State", "Persistent counter.", "int", "7", "0", "100",
                 "editable|serialized|save_game|prefab|replicated", "", "")
    int count{7};

    ARC_PROPERTY("1ef04fe906e7b5bc", "Locked", "State", "Read-only state.", "bool", "false", "", "",
                 "readonly|serialized", "", "")
    bool locked{};
};

} // namespace arc::project::tests
