#pragma once

#include <arc/water/water_types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace arc::water
{

struct water_preset
{
    static constexpr std::uint32_t current_schema_version = 1;

    std::uint32_t schema_version{current_schema_version};
    std::string name{"Open Ocean"};
    water_body_type body_type{water_body_type::ocean};
    water_runtime_settings settings;
};

enum class water_preset_validation_code : std::uint8_t
{
    unsupported_schema,
    missing_name,
    invalid_body_type,
    invalid_simulation,
    invalid_foam,
    invalid_appearance,
    invalid_quality
};

struct water_preset_validation_issue
{
    water_preset_validation_code code{water_preset_validation_code::invalid_simulation};
    std::string message;
};

struct [[nodiscard]] water_preset_validation_result
{
    std::vector<water_preset_validation_issue> issues;

    [[nodiscard]] bool valid() const noexcept
    {
        return issues.empty();
    }
};

[[nodiscard]] water_preset_validation_result validate_water_preset(const water_preset& preset);

} // namespace arc::water
