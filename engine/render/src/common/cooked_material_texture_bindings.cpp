#include <arc/render/cooked_material_texture_bindings.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <utility>

namespace arc::render
{
namespace
{
constexpr std::array<std::byte, 8> signature{
    std::byte{'A'}, std::byte{'R'}, std::byte{'C'}, std::byte{'M'},
    std::byte{'T'}, std::byte{'E'}, std::byte{'X'}, std::byte{'1'}};
constexpr std::uint32_t max_slots = 1024;

void write_u32(std::vector<std::byte>& out, std::uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i)
        out.push_back(static_cast<std::byte>((value >> (i * 8u)) & 0xffu));
}
void write_u64(std::vector<std::byte>& out, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        out.push_back(static_cast<std::byte>((value >> (i * 8u)) & 0xffu));
}
bool read_u32(std::span<const std::byte> bytes, std::size_t& offset, std::uint32_t& value)
{
    if (offset > bytes.size() || bytes.size() - offset < 4) return false;
    value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= std::to_integer<std::uint32_t>(bytes[offset++]) << (i * 8u);
    return true;
}
bool read_u64(std::span<const std::byte> bytes, std::size_t& offset, std::uint64_t& value)
{
    if (offset > bytes.size() || bytes.size() - offset < 8) return false;
    value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= std::to_integer<std::uint64_t>(bytes[offset++]) << (i * 8u);
    return true;
}
bool validate(const cooked_material_texture_manifest& manifest)
{
    if (!manifest.material.valid() || manifest.textures.size() > max_slots) return false;
    std::array<std::uint32_t, 3> next_dimension{};
    std::set<std::uint64_t> parameters;
    for (std::size_t i = 0; i < manifest.textures.size(); ++i)
    {
        const auto& b = manifest.textures[i].binding;
        if (b.slot != i) return false;
        std::size_t dimension{};
        if (b.type == shader_parameter_type::texture_2d)
            dimension = 0;
        else if (b.type == shader_parameter_type::texture_cube)
            dimension = 1;
        else if (b.type == shader_parameter_type::texture_3d)
            dimension = 2;
        else
            return false;
        if (b.dimension_slot != next_dimension[dimension]++) return false;
        if (static_cast<std::uint32_t>(b.semantic) > static_cast<std::uint32_t>(texture_semantic::environment))
            return false;
        if (b.parameter_id.valid() && !parameters.insert(b.parameter_id.representation()).second) return false;
    }
    return true;
}
} // namespace

core::result<std::vector<std::byte>, std::string>
encode_cooked_material_texture_manifest(const cooked_material_texture_manifest& manifest)
{
    using result = core::result<std::vector<std::byte>, std::string>;
    if (!validate(manifest)) return result::failure("Invalid cooked material texture bindings");
    std::vector<std::byte> out(signature.begin(), signature.end());
    write_u64(out, manifest.material.high);
    write_u64(out, manifest.material.low);
    write_u32(out, static_cast<std::uint32_t>(manifest.textures.size()));
    for (const auto& texture : manifest.textures)
    {
        const auto& binding = texture.binding;
        write_u32(out, binding.slot);
        write_u64(out, binding.parameter_id.representation());
        write_u32(out, static_cast<std::uint32_t>(binding.type));
        write_u32(out, binding.dimension_slot);
        write_u32(out, static_cast<std::uint32_t>(binding.semantic));
        write_u64(out, texture.asset.high);
        write_u64(out, texture.asset.low);
    }
    return result::success(std::move(out));
}

core::result<cooked_material_texture_manifest, std::string>
decode_cooked_material_texture_manifest(std::span<const std::byte> bytes)
{
    using result = core::result<cooked_material_texture_manifest, std::string>;
    if (bytes.size() < signature.size() || !std::equal(signature.begin(), signature.end(), bytes.begin()))
        return result::failure("Invalid material texture manifest signature");
    std::size_t offset = signature.size();
    cooked_material_texture_manifest manifest;
    std::uint32_t count{};
    if (!read_u64(bytes, offset, manifest.material.high) || !read_u64(bytes, offset, manifest.material.low) ||
        !read_u32(bytes, offset, count) || count > max_slots ||
        bytes.size() - offset != static_cast<std::size_t>(count) * 40u)
        return result::failure("Invalid material texture manifest header or length");
    manifest.textures.resize(count);
    for (auto& texture : manifest.textures)
    {
        auto& binding = texture.binding;
        std::uint64_t parameter{};
        std::uint32_t type{}, semantic{};
        if (!read_u32(bytes, offset, binding.slot) || !read_u64(bytes, offset, parameter) ||
            !read_u32(bytes, offset, type) || !read_u32(bytes, offset, binding.dimension_slot) ||
            !read_u32(bytes, offset, semantic) || !read_u64(bytes, offset, texture.asset.high) ||
            !read_u64(bytes, offset, texture.asset.low))
            return result::failure("Truncated material texture binding");
        binding.parameter_id = shader_parameter_id{parameter};
        binding.type = static_cast<shader_parameter_type>(type);
        binding.semantic = static_cast<texture_semantic>(semantic);
    }
    if (offset != bytes.size() || !validate(manifest)) return result::failure("Invalid material texture binding slots");
    return result::success(std::move(manifest));
}

core::result<material_descriptor, std::string>
bind_cooked_material_textures(material_descriptor material, const cooked_material_texture_manifest& manifest,
                              std::span<const std::pair<assets::asset_guid, texture_handle>> resident)
{
    using result = core::result<material_descriptor, std::string>;
    if (!material.runtime_program || material.shader_package != manifest.material || !validate(manifest))
        return result::failure("Material texture manifest identity or layout mismatch");

    std::vector<texture_handle> handles;
    handles.reserve(manifest.textures.size());
    for (const auto& entry : manifest.textures)
    {
        if (entry.binding.parameter_id.valid())
        {
            const auto& parameters = material.runtime_program->parameters;
            const auto match = std::find_if(parameters.begin(), parameters.end(), [&](const auto& parameter)
                                            {
                                                return parameter.id == entry.binding.parameter_id &&
                                                       parameter.type == entry.binding.type;
                                            });
            if (match == parameters.end())
                return result::failure("Material texture binding has incompatible reflected parameter");
        }
        texture_handle handle{};
        if (entry.asset.valid())
        {
            const auto found = std::find_if(resident.begin(), resident.end(),
                                            [&](const auto& item) { return item.first == entry.asset; });
            if (found == resident.end() || !found->second.valid())
                return result::failure("Cooked material references a nonresident texture GUID");
            handle = found->second;
        }
        handles.push_back(handle);
    }

    auto runtime = std::make_shared<material_runtime_program>(*material.runtime_program);
    runtime->texture_bindings.reserve(manifest.textures.size());
    for (const auto& entry : manifest.textures)
        runtime->texture_bindings.push_back(entry.binding);
    material.runtime_program = std::move(runtime);
    material.runtime_textures = std::move(handles);
    return result::success(std::move(material));
}
} // namespace arc::render
