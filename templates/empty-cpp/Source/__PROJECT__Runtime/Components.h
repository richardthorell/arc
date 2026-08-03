#pragma once
#include <arc/project/reflection.h>
namespace {{PROJECT_TOKEN}} {
ARC_COMPONENT("8ab7fc20dce64be2843a680179249a5f", 1, "Project Component", "Gameplay", "Example reflected project component.")
struct project_component {
  ARC_PROPERTY("172b6e2288c9ef10", "Value", "General", "Example authored value.", "float", "1.0", "0", "100",
               "editable|serialized|prefab", "", "")
  float value{1.0F};
}; }
