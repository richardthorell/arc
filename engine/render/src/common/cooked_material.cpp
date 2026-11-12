#include <arc/render/cooked_material.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>

namespace arc::render
{
namespace
{
class reader
{
public:
    explicit reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <class T> bool value(T& output)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (cursor_ > bytes_.size() || sizeof(T) > bytes_.size() - cursor_) return false;
        std::memcpy(&output, bytes_.data() + cursor_, sizeof(T));
        cursor_ += sizeof(T);
        return true;
    }

    bool string(std::string& output)
    {
        std::uint64_t count{};
        if (!value(count) || count > bytes_.size() - cursor_ || count > max_string_bytes) return false;
        output.assign(reinterpret_cast<const char*>(bytes_.data() + cursor_), static_cast<std::size_t>(count));
        cursor_ += static_cast<std::size_t>(count);
        return true;
    }

    bool bytes(std::span<std::byte> output)
    {
        if (cursor_ > bytes_.size() || output.size() > bytes_.size() - cursor_) return false;
        std::memcpy(output.data(), bytes_.data() + cursor_, output.size());
        cursor_ += output.size();
        return true;
    }

    [[nodiscard]] bool complete() const noexcept
    {
        return cursor_ == bytes_.size();
    }

private:
    static constexpr std::uint64_t max_string_bytes = 8u * 1024u * 1024u;
    std::span<const std::byte> bytes_;
    std::size_t cursor_{};
};

bool valid_pass(material_pass pass) noexcept
{
    switch (pass)
    {
        case material_pass::depth:
        case material_pass::shadow:
        case material_pass::gbuffer:
        case material_pass::forward:
        case material_pass::motion:
        case material_pass::object_id:
        case material_pass::selection:
        case material_pass::ray_hit:
            return true;
    }
    return false;
}
} // namespace

core::result<cooked_material_metadata, std::string> decode_cooked_material_metadata(std::span<const std::byte> bytes)
{
    using result = core::result<cooked_material_metadata, std::string>;
    reader input(bytes);
    cooked_material_metadata material;
    std::string signature;
    if (!input.string(signature) || signature != "ARC_MATERIAL_4")
        return result::failure("Invalid ARC_MATERIAL_4 signature");
    if (!input.value(material.compiled.contract_version) || !input.value(material.compiled.material_abi))
        return result::failure("Truncated ARC_MATERIAL_4 contract");
    if (!input.value(material.compiled.package.high) || !input.value(material.compiled.package.low))
        return result::failure("Truncated ARC_MATERIAL_4 package identity");
    if (material.compiled.contract_version != material_pass_contract_version ||
        material.compiled.material_abi != material_abi_version)
        return result::failure("Incompatible material pass contract or ABI");

    std::uint32_t count{};
    if (!input.value(count) || count > 32 || (count > 0 && !material.compiled.package.valid()))
        return result::failure("Invalid cooked material pass count/package ID");
    std::set<std::pair<material_pass, std::uint64_t>> unique_passes;
    for (std::uint32_t i = 0; i < count; ++i)
    {
        material_pass_binding binding;
        std::uint64_t permutation{}, entry_point{};
        if (!input.value(binding.pass) || !input.value(permutation) || !input.value(entry_point))
            return result::failure("Truncated cooked material pass identity");
        if (!input.bytes(binding.build_hash.bytes)) return result::failure("Truncated cooked material pass hash");
        if (!valid_pass(binding.pass) || permutation == 0 || entry_point == 0)
            return result::failure("Invalid cooked material pass identity");
        if (!unique_passes.emplace(binding.pass, permutation).second)
            return result::failure("Duplicate cooked material pass");
        binding.permutation = shader_permutation_id{permutation};
        binding.entry_point = shader_entry_point_id{entry_point};
        material.compiled.passes.push_back(std::move(binding));
    }

    if (!input.value(count) || count > 65536u) return result::failure("Invalid cooked material parameter count");
    std::set<std::uint64_t> unique_parameters;
    for (std::uint32_t i = 0; i < count; ++i)
    {
        shader_parameter_descriptor parameter;
        std::uint64_t id{};
        if (!input.value(id) || !input.string(parameter.name) || !input.value(parameter.type) ||
            !input.value(parameter.offset) || !input.value(parameter.size) || !input.value(parameter.has_range) ||
            !input.value(parameter.minimum) || !input.value(parameter.maximum))
            return result::failure("Truncated cooked material parameter");
        if (id == 0 || !unique_parameters.insert(id).second)
            return result::failure("Invalid or duplicate cooked material parameter ID");
        if (!std::isfinite(parameter.minimum) || !std::isfinite(parameter.maximum) ||
            (parameter.has_range && parameter.minimum > parameter.maximum) ||
            parameter.size > std::numeric_limits<std::uint32_t>::max() - parameter.offset)
            return result::failure("Invalid cooked material parameter range");
        const auto parameter_type_id = static_cast<std::uint32_t>(parameter.type);
        if (parameter_type_id > static_cast<std::uint32_t>(shader_parameter_type::texture_3d))
            return result::failure("Invalid cooked material parameter type");
        parameter.id = shader_parameter_id{id};
        material.parameters.push_back(std::move(parameter));
    }

    if (!input.string(material.canonical_document_json) || !input.complete() ||
        material.canonical_document_json.empty())
        return result::failure("Invalid cooked material document/trailing bytes");
    // Terrain materials can legitimately omit executable passes.
    // The existing cooker/authoring contract validates the canonical JSON.
    return result::success(std::move(material));
}
} // namespace arc::render
