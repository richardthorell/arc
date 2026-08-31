#pragma once

#include <arc/assets/cook.h>
#include <arc/render/texture_artifact.h>

#include <optional>
#include <string>
#include <string_view>

namespace arc::render::tools
{

/** @brief High-level authored texture defaults. Applying a preset writes concrete settings. */
enum class texture_import_preset : std::uint8_t
{
    custom,
    color,
    normal_map,
    data,
    hdr,
    ui,
    environment
};

/** @brief Versioned texture settings stored in a texture source's `.arcmeta` sidecar. */
struct texture_import_settings
{
    static constexpr std::uint32_t current_version = 3;
    texture_import_preset preset{texture_import_preset::custom};
    texture_semantic semantic{texture_semantic::generic_color};
    texture_color_space color_space{texture_color_space::srgb};
    texture_streaming_mode streaming_mode{texture_streaming_mode::resident};
};

using texture_import_settings_result = core::result<texture_import_settings, std::string>;

[[nodiscard]] std::string_view texture_import_preset_name(texture_import_preset preset) noexcept;
[[nodiscard]] std::string_view texture_semantic_name(texture_semantic semantic) noexcept;
[[nodiscard]] std::string_view texture_color_space_name(texture_color_space color_space) noexcept;
[[nodiscard]] std::string_view texture_streaming_mode_name(texture_streaming_mode mode) noexcept;
[[nodiscard]] std::optional<texture_import_preset> parse_texture_import_preset(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_semantic> parse_texture_semantic(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_color_space> parse_texture_color_space(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_streaming_mode> parse_texture_streaming_mode(std::string_view value) noexcept;

/** @brief Resolve a user-facing preset to explicit authored defaults. */
[[nodiscard]] texture_import_settings texture_import_settings_for_preset(texture_import_preset preset) noexcept;

/** @brief Parse legacy and current settings, filling absent fields with compatible defaults. */
[[nodiscard]] texture_import_settings_result
parse_texture_import_settings(std::string_view canonical_json,
                              std::uint32_t settings_version = texture_import_settings::current_version);

/** @brief Serialize canonical v3 settings for `.arcmeta`. */
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
