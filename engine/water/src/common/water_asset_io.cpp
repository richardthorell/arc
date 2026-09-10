#include <arc/water/water_asset_io.h>

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace arc::water
{
namespace
{

using json = nlohmann::json;
constexpr std::uint32_t water_document_format_version = 1;

template <class Result> Result failure(water_asset_io_error_code code, std::string message)
{
    return Result::failure({code, std::move(message)});
}

const char* body_type_name(water_body_type value) noexcept
{
    switch (value)
    {
        case water_body_type::ocean:
            return "ocean";
        case water_body_type::lake:
            return "lake";
        case water_body_type::river:
            return "river";
    }
    return "ocean";
}

std::optional<water_body_type> parse_body_type(std::string_view value) noexcept
{
    if (value == "ocean") return water_body_type::ocean;
    if (value == "lake") return water_body_type::lake;
    if (value == "river") return water_body_type::river;
    return std::nullopt;
}

const char* quality_name(water_quality value) noexcept
{
    switch (value)
    {
        case water_quality::low:
            return "low";
        case water_quality::medium:
            return "medium";
        case water_quality::high:
            return "high";
        case water_quality::ultra:
            return "ultra";
    }
    return "high";
}

std::optional<water_quality> parse_quality(std::string_view value) noexcept
{
    if (value == "low") return water_quality::low;
    if (value == "medium") return water_quality::medium;
    if (value == "high") return water_quality::high;
    if (value == "ultra") return water_quality::ultra;
    return std::nullopt;
}

class water_asset_importer final : public assets::asset_importer
{
public:
    water_asset_importer()
    {
        descriptor_.id = assets::importer_ids::water_preset;
        descriptor_.name = "ARC Water Preset";
        descriptor_.version = 1;
        descriptor_.settings_version = 1;
        descriptor_.extensions = {".arcwater"};
        descriptor_.output_types = {assets::asset_types::water_preset};
    }

    const assets::asset_importer_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }

    assets::asset_import_result import(const assets::asset_import_context& context) override
    {
        if (context.cancellation.stop_requested())
            return {.error = {.code = assets::asset_error_code::cancelled,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = "Water preset import was cancelled"}};

        const std::string text(reinterpret_cast<const char*>(context.source_bytes.data()), context.source_bytes.size());
        auto decoded = read_water_asset_json(text);
        if (!decoded)
            return {.error = {.code = assets::asset_error_code::import_failed,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = decoded.error().message}};

        auto preset = std::make_shared<water_preset>(std::move(decoded.value()));
        assets::asset_import_result result;
        result.payload = assets::asset_payload::make<water_preset>(
            assets::asset_types::water_preset, std::move(preset), sizeof(water_preset) + context.source_bytes.size());
        return result;
    }

private:
    assets::asset_importer_descriptor descriptor_;
};

} // namespace

water_asset_json_result write_water_asset_json(const water_preset& preset, bool pretty)
{
    if (!validate_water_preset(preset).valid())
        return failure<water_asset_json_result>(water_asset_io_error_code::invalid_asset,
                                                "Cannot serialize an invalid Water preset");

    const auto& simulation = preset.settings.simulation;
    const auto& foam = preset.settings.foam;
    const auto& appearance = preset.settings.appearance;
    const json document{
        {"format", "arc.water-preset"},
        {"formatVersion", water_document_format_version},
        {"preset",
         {{"schemaVersion", preset.schema_version},
          {"name", preset.name},
          {"bodyType", body_type_name(preset.body_type)},
          {"simulation",
           {{"windSpeed", simulation.wind_speed},
            {"windDirection", {simulation.wind_direction[0], simulation.wind_direction[1]}},
            {"fetchLength", simulation.fetch_length},
            {"waveAmplitude", simulation.wave_amplitude},
            {"choppiness", simulation.choppiness},
            {"seed", simulation.seed}}},
          {"foam", {{"enabled", foam.enabled}, {"threshold", foam.threshold}, {"decay", foam.decay}}},
          {"appearance",
           {{"absorption", {appearance.absorption[0], appearance.absorption[1], appearance.absorption[2]}},
            {"scattering", {appearance.scattering[0], appearance.scattering[1], appearance.scattering[2]}},
            {"roughness", appearance.roughness},
            {"refractionStrength", appearance.refraction_strength}}},
          {"quality", quality_name(preset.settings.quality)}}}};
    return water_asset_json_result::success(document.dump(pretty ? 2 : -1) + (pretty ? "\n" : ""));
}

water_asset_decode_result read_water_asset_json(std::string_view text)
{
    try
    {
        const auto document = json::parse(text);
        if (!document.is_object() || document.value("format", "") != "arc.water-preset" ||
            !document.contains("formatVersion") || !document["formatVersion"].is_number_unsigned() ||
            !document.contains("preset") || !document["preset"].is_object())
            return failure<water_asset_decode_result>(water_asset_io_error_code::invalid_document,
                                                       "Invalid Water preset document");
        if (document["formatVersion"].get<std::uint32_t>() != water_document_format_version)
            return failure<water_asset_decode_result>(water_asset_io_error_code::unsupported_format_version,
                                                       "Unsupported Water preset format version");

        const auto& value = document["preset"];
        water_preset preset;
        preset.schema_version = value.at("schemaVersion").get<std::uint32_t>();
        if (preset.schema_version != water_preset::current_schema_version)
            return failure<water_asset_decode_result>(water_asset_io_error_code::unsupported_schema_version,
                                                       "Unsupported Water preset schema version");
        preset.name = value.at("name").get<std::string>();
        const auto body_type = parse_body_type(value.at("bodyType").get<std::string>());
        const auto quality = parse_quality(value.at("quality").get<std::string>());
        if (!body_type || !quality)
            return failure<water_asset_decode_result>(water_asset_io_error_code::invalid_document,
                                                       "Water preset enum value is invalid");
        preset.body_type = *body_type;
        preset.settings.quality = *quality;

        const auto& simulation = value.at("simulation");
        const auto direction = simulation.at("windDirection").get<std::vector<float>>();
        if (direction.size() != 2)
            return failure<water_asset_decode_result>(water_asset_io_error_code::invalid_document,
                                                       "Water wind direction must have two components");
        preset.settings.simulation.wind_speed = simulation.at("windSpeed").get<float>();
        preset.settings.simulation.wind_direction = {direction[0], direction[1]};
        preset.settings.simulation.fetch_length = simulation.at("fetchLength").get<float>();
        preset.settings.simulation.wave_amplitude = simulation.at("waveAmplitude").get<float>();
        preset.settings.simulation.choppiness = simulation.at("choppiness").get<float>();
        preset.settings.simulation.seed = simulation.at("seed").get<std::uint64_t>();

        const auto& foam = value.at("foam");
        preset.settings.foam.enabled = foam.at("enabled").get<bool>();
        preset.settings.foam.threshold = foam.at("threshold").get<float>();
        preset.settings.foam.decay = foam.at("decay").get<float>();

        const auto& appearance = value.at("appearance");
        const auto absorption = appearance.at("absorption").get<std::vector<float>>();
        const auto scattering = appearance.at("scattering").get<std::vector<float>>();
        if (absorption.size() != 3 || scattering.size() != 3)
            return failure<water_asset_decode_result>(water_asset_io_error_code::invalid_document,
                                                       "Water optical vectors must have three components");
        preset.settings.appearance.absorption = {absorption[0], absorption[1], absorption[2]};
        preset.settings.appearance.scattering = {scattering[0], scattering[1], scattering[2]};
        preset.settings.appearance.roughness = appearance.at("roughness").get<float>();
        preset.settings.appearance.refraction_strength = appearance.at("refractionStrength").get<float>();

        if (!validate_water_preset(preset).valid())
            return failure<water_asset_decode_result>(water_asset_io_error_code::invalid_asset,
                                                       "Water preset values are invalid");
        return water_asset_decode_result::success(std::move(preset));
    }
    catch (const json::exception& error)
    {
        return failure<water_asset_decode_result>(water_asset_io_error_code::invalid_document,
                                                   std::string{"Invalid Water preset JSON: "} + error.what());
    }
}

std::unique_ptr<assets::asset_importer> make_water_asset_importer()
{
    return std::make_unique<water_asset_importer>();
}

bool register_water_asset_importer(assets::asset_manager& manager)
{
    return manager.register_importer(make_water_asset_importer());
}

} // namespace arc::water
