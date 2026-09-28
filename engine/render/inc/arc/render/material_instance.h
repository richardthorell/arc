#pragma once

#include <arc/render/material.h>

#include <cstdint>
#include <string>
#include <unordered_set>

namespace arc::render
{

enum class material_instance_validation_error : std::uint8_t
{
    none,
    missing_parent,
    invalid_parameter_id,
    duplicate_parameter_id
};

struct [[nodiscard]] material_instance_validation_result
{
    material_instance_validation_error error{material_instance_validation_error::none};
    shader_parameter_id parameter_id{};
    std::string message;

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return error == material_instance_validation_error::none;
    }
};

/**
 * @brief Validate material-instance invariants that do not require resolving the parent material.
 *
 * Parent parameter existence and type compatibility are checked by resolve_material_instance(),
 * where the parent definition is available.
 */
[[nodiscard]] inline material_instance_validation_result
validate_material_instance(const material_instance_descriptor& instance)
{
    if (!instance.parent.valid())
    {
        return {material_instance_validation_error::missing_parent,
                {},
                "material instance requires a valid parent material"};
    }

    std::unordered_set<std::uint64_t> parameter_ids;
    parameter_ids.reserve(instance.overrides.size());
    for (const auto& override_value : instance.overrides)
    {
        if (!override_value.id.valid())
        {
            return {material_instance_validation_error::invalid_parameter_id, override_value.id,
                    "material instance override requires a valid parameter id"};
        }
        if (!parameter_ids.insert(override_value.id.value).second)
        {
            return {material_instance_validation_error::duplicate_parameter_id, override_value.id,
                    "material instance cannot override the same parameter more than once"};
        }
    }

    return {};
}

} // namespace arc::render
