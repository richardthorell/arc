#pragma once

#include <arc/assets/cook.h>
#include <arc/render/texture_artifact.h>

#include <optional>
#include <string>
#include <string_view>

namespace arc::render::tools
{

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

enum class texture_mip_generation_filter : std::uint8_t
{
    box,
    nearest
};

struct texture_import_settings
{
    static constexpr std::uint32_t current_version = 4;
    texture_import_preset preset{texture_import_preset::custom};
    texture_semantic semantic{texture_semantic::generic_color};
    texture_color_space color_space{texture_color_space::srgb};
    texture_streaming_mode streaming_mode{texture_streaming_mode::resident};
    texture_compression_policy compression{texture_compression_policy::automatic};
    texture_power_of_two_policy power_of_two{texture_power_of_two_policy::preserve};
    texture_filter_mode min_filter{texture_filter_mode::linear};
    texture_filter_mode mag_filter{texture_filter_mode::linear};
    texture_mip_filter_mode mip_filter{texture_mip_filter_mode::linear};
    texture_address_mode wrap_u{texture_address_mode::repeat};
    texture_address_mode wrap_v{texture_address_mode::repeat};
    texture_mip_generation_filter mip_generation_filter{texture_mip_generation_filter::box};
    std::uint32_t max_size{8192};
    float anisotropy{8.0f};
    float lod_bias{};
    float minimum_lod{};
    float maximum_lod{1000.0f};
    float alpha_coverage_threshold{0.5f};
    bool generate_mips{true};
    bool preserve_alpha_coverage{};
};

struct [[nodiscard]] texture_preprocess_result
{
    texture_data texture;
    texture_artifact_metadata metadata;
    std::vector<assets::asset_diagnostic> diagnostics;
};

using texture_import_settings_result = core::result<texture_import_settings, std::string>;
using texture_preprocess_result_type = core::result<texture_preprocess_result, std::string>;

[[nodiscard]] std::string_view texture_import_preset_name(texture_import_preset preset) noexcept;
[[nodiscard]] std::string_view texture_semantic_name(texture_semantic semantic) noexcept;
[[nodiscard]] std::string_view texture_color_space_name(texture_color_space color_space) noexcept;
[[nodiscard]] std::string_view texture_streaming_mode_name(texture_streaming_mode mode) noexcept;
[[nodiscard]] std::string_view texture_compression_policy_name(texture_compression_policy policy) noexcept;
[[nodiscard]] std::string_view texture_power_of_two_policy_name(texture_power_of_two_policy policy) noexcept;
[[nodiscard]] std::string_view texture_filter_mode_name(texture_filter_mode filter) noexcept;
[[nodiscard]] std::string_view texture_mip_filter_mode_name(texture_mip_filter_mode filter) noexcept;
[[nodiscard]] std::string_view texture_address_mode_name(texture_address_mode mode) noexcept;
[[nodiscard]] std::string_view texture_mip_generation_filter_name(texture_mip_generation_filter filter) noexcept;
[[nodiscard]] std::optional<texture_import_preset> parse_texture_import_preset(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_semantic> parse_texture_semantic(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_color_space> parse_texture_color_space(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_streaming_mode> parse_texture_streaming_mode(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_compression_policy>
parse_texture_compression_policy(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_power_of_two_policy>
parse_texture_power_of_two_policy(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_filter_mode> parse_texture_filter_mode(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_mip_filter_mode> parse_texture_mip_filter_mode(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_address_mode> parse_texture_address_mode(std::string_view value) noexcept;
[[nodiscard]] std::optional<texture_mip_generation_filter>
parse_texture_mip_generation_filter(std::string_view value) noexcept;

[[nodiscard]] texture_import_settings texture_import_settings_for_preset(texture_import_preset preset) noexcept;
[[nodiscard]] texture_import_settings_result
parse_texture_import_settings(std::string_view canonical_json,
                              std::uint32_t settings_version = texture_import_settings::current_version);
[[nodiscard]] std::string serialize_texture_import_settings(const texture_import_settings& settings);
[[nodiscard]] texture_preprocess_result_type preprocess_texture_for_cook(texture_data texture,
                                                                         const texture_import_settings& settings,
                                                                         const assets::cook_target& target);

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
