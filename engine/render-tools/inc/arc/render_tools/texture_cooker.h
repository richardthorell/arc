#pragma once

#include <arc/assets/cook.h>
#include <arc/render/texture_artifact.h>

#include <string>
#include <string_view>

namespace arc::render::tools
{

/** @brief Versioned texture settings stored in a texture source's `.arcmeta` sidecar. */
struct texture_import_settings
{
    static constexpr std::uint32_t current_version = 2;
    texture_streaming_mode streaming_mode{texture_streaming_mode::resident};
};

using texture_import_settings_result = core::result<texture_import_settings, std::string>;

/** @brief Parse v1/v2 settings; absent legacy fields migrate to Resident. */
[[nodiscard]] texture_import_settings_result parse_texture_import_settings(std::string_view canonical_json,
                                                                           std::uint32_t settings_version = 2);

/** @brief Serialize canonical v2 settings for `.arcmeta`. */
[[nodiscard]] std::string serialize_texture_import_settings(const texture_import_settings& settings);

/** @brief Cook DDS/native or decoded RGBA8 sources into range-readable `.arctex` v1 artifacts. */
class texture_cook_processor final : public assets::asset_cook_processor
{
public:
    texture_cook_processor();
    [[nodiscard]] const assets::asset_cook_processor_descriptor& descriptor() const noexcept override;
    [[nodiscard]] std::string toolchain_fingerprint() const override;
    [[nodiscard]] assets::asset_cook_result cook(const assets::asset_cook_context& context) override;

private:
    assets::asset_cook_processor_descriptor descriptor_;
};

} // namespace arc::render::tools
