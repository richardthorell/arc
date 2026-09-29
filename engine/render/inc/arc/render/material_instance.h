#pragma once

#include <arc/render/material.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>

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

/** @brief Find an authored override by stable reflected parameter identity. */
[[nodiscard]] inline const material_parameter_override*
find_material_instance_override(const material_instance_descriptor& instance, shader_parameter_id parameter_id) noexcept
{
    const auto it = std::find_if(instance.overrides.begin(), instance.overrides.end(),
                                 [parameter_id](const material_parameter_override& override_value)
                                 { return override_value.id == parameter_id; });
    return it == instance.overrides.end() ? nullptr : &*it;
}

/** @brief Return whether the instance currently overrides a reflected parent parameter. */
[[nodiscard]] inline bool is_material_instance_parameter_overridden(const material_instance_descriptor& instance,
                                                                    shader_parameter_id parameter_id) noexcept
{
    return find_material_instance_override(instance, parameter_id) != nullptr;
}

/**
 * @brief Add or replace one parameter override while preserving deterministic authored order.
 *
 * Replacing an existing override keeps its position. New overrides append in authoring order, so editor reset/reapply
 * operations do not reorder unrelated parameters or require a second mutation representation.
 */
inline void set_material_instance_override(material_instance_descriptor& instance, material_parameter_override value)
{
    const auto it =
        std::find_if(instance.overrides.begin(), instance.overrides.end(),
                     [&value](const material_parameter_override& existing) { return existing.id == value.id; });
    if (it != instance.overrides.end())
    {
        *it = std::move(value);
        return;
    }

    instance.overrides.push_back(std::move(value));
}

/** @brief Reset one parameter to its parent value. Returns true when an authored override was removed. */
inline bool reset_material_instance_override(material_instance_descriptor& instance, shader_parameter_id parameter_id)
{
    const auto it = std::find_if(instance.overrides.begin(), instance.overrides.end(),
                                 [parameter_id](const material_parameter_override& override_value)
                                 { return override_value.id == parameter_id; });
    if (it == instance.overrides.end())
    {
        return false;
    }

    instance.overrides.erase(it);
    return true;
}

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
