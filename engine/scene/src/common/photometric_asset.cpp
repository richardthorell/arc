#include <arc/scene/photometric_asset.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>

namespace arc::scene
{
namespace
{
constexpr std::uint32_t artifact_magic = 0x53454941u;
constexpr std::uint32_t artifact_version = 1u;
constexpr std::uint32_t maximum_angle_count = 65536u;
constexpr std::uint64_t maximum_sample_count = 16ull * 1024ull * 1024ull;
constexpr std::size_t header_bytes = 8u * sizeof(std::uint32_t);

void append_u32(std::vector<std::byte>& output, std::uint32_t value)
{
    output.push_back(static_cast<std::byte>(value & 0xffu));
    output.push_back(static_cast<std::byte>((value >> 8u) & 0xffu));
    output.push_back(static_cast<std::byte>((value >> 16u) & 0xffu));
    output.push_back(static_cast<std::byte>((value >> 24u) & 0xffu));
}

void append_f32(std::vector<std::byte>& output, float value)
{
    append_u32(output, std::bit_cast<std::uint32_t>(value));
}

bool read_u32(std::span<const std::byte> bytes, std::size_t& offset, std::uint32_t& value) noexcept
{
    if (offset + 4u > bytes.size()) return false;
    value = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset])) |
            (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset + 1u])) << 8u) |
            (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset + 2u])) << 16u) |
            (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[offset + 3u])) << 24u);
    offset += 4u;
    return true;
}

bool read_f32(std::span<const std::byte> bytes, std::size_t& offset, float& value) noexcept
{
    std::uint32_t bits{};
    if (!read_u32(bytes, offset, bits)) return false;
    value = std::bit_cast<float>(bits);
    return true;
}

bool valid_profile(const render::photometric_profile& profile) noexcept
{
    if (profile.vertical_angles_degrees.empty() || profile.horizontal_angles_degrees.empty()) return false;
    const std::uint64_t samples = static_cast<std::uint64_t>(profile.vertical_angles_degrees.size()) *
                                  static_cast<std::uint64_t>(profile.horizontal_angles_degrees.size());
    if (samples != profile.normalized_candela.size() || samples > maximum_sample_count) return false;
    if (profile.vertical_angles_degrees.size() > maximum_angle_count ||
        profile.horizontal_angles_degrees.size() > maximum_angle_count)
        return false;
    if (profile.photometric_type < 1u || profile.photometric_type > 3u ||
        !std::isfinite(profile.peak_candela) || profile.peak_candela <= 0.0f ||
        !std::isfinite(profile.declared_lumens) || profile.declared_lumens < 0.0f)
        return false;
    if (!std::is_sorted(profile.vertical_angles_degrees.begin(), profile.vertical_angles_degrees.end()) ||
        !std::is_sorted(profile.horizontal_angles_degrees.begin(), profile.horizontal_angles_degrees.end()))
        return false;
    const auto finite = [](float value) { return std::isfinite(value); };
    return std::all_of(profile.vertical_angles_degrees.begin(), profile.vertical_angles_degrees.end(), finite) &&
           std::all_of(profile.horizontal_angles_degrees.begin(), profile.horizontal_angles_degrees.end(), finite) &&
           std::all_of(profile.normalized_candela.begin(), profile.normalized_candela.end(),
                       [](float value) { return std::isfinite(value) && value >= 0.0f && value <= 1.0f; });
}

class photometric_profile_importer final : public assets::asset_importer
{
public:
    photometric_profile_importer()
    {
        descriptor_.id = assets::importer_ids::photometric_profile;
        descriptor_.name = "IES Photometric Profile";
        descriptor_.version = 1u;
        descriptor_.settings_version = 1u;
        descriptor_.extensions = {".ies"};
        descriptor_.output_types = {assets::asset_types::photometric_profile};
    }

    const assets::asset_importer_descriptor& descriptor() const noexcept override { return descriptor_; }

    assets::asset_import_result import(const assets::asset_import_context& context) override
    {
        if (context.cancellation.stop_requested())
            return {.error = {.code = assets::asset_error_code::cancelled,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = "IES profile import was cancelled"}};

        const std::string source(reinterpret_cast<const char*>(context.source_bytes.data()), context.source_bytes.size());
        auto parsed = render::parse_ies_profile(source);
        if (!parsed)
            return {.error = {.code = assets::asset_error_code::import_failed,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = parsed.error}};

        auto profile = std::make_shared<render::photometric_profile>(std::move(parsed.profile));
        auto artifact = encode_photometric_profile_artifact(*profile);
        if (artifact.empty())
            return {.error = {.code = assets::asset_error_code::import_failed,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = "IES profile produced an invalid cooked representation"}};

        const std::size_t resident_bytes =
            sizeof(render::photometric_profile) +
            (profile->vertical_angles_degrees.size() + profile->horizontal_angles_degrees.size() +
             profile->normalized_candela.size()) *
                sizeof(float);

        assets::asset_import_result result;
        result.payload = assets::asset_payload::make<render::photometric_profile>(
            assets::asset_types::photometric_profile, std::move(profile), resident_bytes);
        result.artifacts.push_back({.name = "photometric",
                                    .extension = ".arcies",
                                    .bytes = std::move(artifact),
                                    .residency = assets::asset_residency::derived});
        result.residency = assets::asset_residency::cpu;
        return result;
    }

private:
    assets::asset_importer_descriptor descriptor_;
};

} // namespace

std::vector<std::byte> encode_photometric_profile_artifact(const render::photometric_profile& profile)
{
    if (!valid_profile(profile)) return {};
    const auto vertical_count = static_cast<std::uint32_t>(profile.vertical_angles_degrees.size());
    const auto horizontal_count = static_cast<std::uint32_t>(profile.horizontal_angles_degrees.size());
    const auto sample_count = static_cast<std::uint32_t>(profile.normalized_candela.size());

    std::vector<std::byte> output;
    output.reserve(header_bytes + static_cast<std::size_t>(vertical_count + horizontal_count + sample_count) * 4u);
    append_u32(output, artifact_magic);
    append_u32(output, artifact_version);
    append_u32(output, vertical_count);
    append_u32(output, horizontal_count);
    append_u32(output, sample_count);
    append_u32(output, profile.photometric_type);
    append_f32(output, profile.peak_candela);
    append_f32(output, profile.declared_lumens);
    for (float value : profile.vertical_angles_degrees) append_f32(output, value);
    for (float value : profile.horizontal_angles_degrees) append_f32(output, value);
    for (float value : profile.normalized_candela) append_f32(output, value);
    return output;
}

photometric_profile_artifact_result decode_photometric_profile_artifact(std::span<const std::byte> bytes)
{
    const auto failure = [](std::string message)
    { return photometric_profile_artifact_result{.error = std::move(message)}; };
    if (bytes.size() < header_bytes) return failure("Cooked IES profile is smaller than its header");

    std::size_t offset{};
    std::uint32_t magic{}, version{}, vertical_count{}, horizontal_count{}, sample_count{}, photometric_type{};
    float peak_candela{}, declared_lumens{};
    if (!read_u32(bytes, offset, magic) || !read_u32(bytes, offset, version) ||
        !read_u32(bytes, offset, vertical_count) || !read_u32(bytes, offset, horizontal_count) ||
        !read_u32(bytes, offset, sample_count) || !read_u32(bytes, offset, photometric_type) ||
        !read_f32(bytes, offset, peak_candela) || !read_f32(bytes, offset, declared_lumens))
        return failure("Cooked IES profile header is truncated");

    if (magic != artifact_magic) return failure("Cooked IES profile has invalid magic");
    if (version != artifact_version) return failure("Cooked IES profile version is unsupported");
    if (vertical_count == 0u || horizontal_count == 0u || vertical_count > maximum_angle_count ||
        horizontal_count > maximum_angle_count)
        return failure("Cooked IES profile angle dimensions are invalid");

    const std::uint64_t expected_samples = static_cast<std::uint64_t>(vertical_count) * horizontal_count;
    if (expected_samples != sample_count || expected_samples > maximum_sample_count)
        return failure("Cooked IES profile sample dimensions are invalid");

    const std::size_t expected_bytes =
        header_bytes + static_cast<std::size_t>(vertical_count + horizontal_count + sample_count) * 4u;
    if (bytes.size() != expected_bytes) return failure("Cooked IES profile byte size does not match its header");

    render::photometric_profile profile;
    profile.photometric_type = photometric_type;
    profile.peak_candela = peak_candela;
    profile.declared_lumens = declared_lumens;
    profile.vertical_angles_degrees.resize(vertical_count);
    profile.horizontal_angles_degrees.resize(horizontal_count);
    profile.normalized_candela.resize(sample_count);
    for (auto& value : profile.vertical_angles_degrees)
        if (!read_f32(bytes, offset, value)) return failure("Cooked IES vertical-angle table is truncated");
    for (auto& value : profile.horizontal_angles_degrees)
        if (!read_f32(bytes, offset, value)) return failure("Cooked IES horizontal-angle table is truncated");
    for (auto& value : profile.normalized_candela)
        if (!read_f32(bytes, offset, value)) return failure("Cooked IES candela table is truncated");

    if (!valid_profile(profile)) return failure("Cooked IES profile failed validation");
    return {.profile = std::move(profile)};
}

std::unique_ptr<assets::asset_importer> make_photometric_profile_importer()
{
    return std::make_unique<photometric_profile_importer>();
}

bool register_photometric_profile_importer(assets::asset_manager& manager)
{
    return manager.register_importer(make_photometric_profile_importer());
}

} // namespace arc::scene
