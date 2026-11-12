#include <arc/render/cooked_material.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string_view>
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
        material.compiled.passes.push_back(binding);
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

core::result<material_runtime_program, std::string>
realize_cooked_material_program(const cooked_material_metadata& material,
                                std::span<const cooked_material_shader> shaders)
{
    using result = core::result<material_runtime_program, std::string>;
    if (material.compiled.contract_version != material_pass_contract_version ||
        material.compiled.material_abi != material_abi_version || !material.compiled.package.valid())
        return result::failure("Cooked material has an incompatible runtime pass contract");

    material_runtime_program program;
    program.parameters = material.parameters;
    if (material.compiled.passes.empty()) return result::failure("Cooked material has no executable shader passes");

    for (const auto& binding : material.compiled.passes)
    {
        const cooked_material_shader* matched{};
        for (const auto& shader : shaders)
        {
            if (shader.package.id != material.compiled.package || shader.package.permutation != binding.permutation ||
                shader.package.compiled.build_hash != binding.build_hash)
                continue;
            if (matched) return result::failure("Ambiguous cooked shader package for material pass");
            matched = &shader;
        }
        if (!matched) return result::failure("Missing cooked shader package for material pass");

        const auto& package = matched->package;
        if (package.target != shader_target::spirv || !package.generation.valid() || package.compiled.bytecode.empty())
            return result::failure("Cooked shader package has no executable SPIR-V payload");
        const auto& reflection = package.compiled.reflection;
        const auto entry = std::find_if(reflection.entry_points.begin(), reflection.entry_points.end(),
                                        [&](const auto& value) { return value.id == binding.entry_point; });
        if (entry == reflection.entry_points.end() || entry->stage != shader_stage::fragment)
            return result::failure("Cooked material pass has no matching fragment entry point");
        bool reflected_pass = false;
        for (const auto& value : reflection.passes)
            if (value.pass == binding.pass && value.entry_point == binding.entry_point) reflected_pass = true;
        if (!reflected_pass)
            return result::failure("Cooked shader reflection does not implement the declared material pass");
        if (reflection.parameters.size() != material.parameters.size())
            return result::failure("Cooked shader parameter count differs from material contract");
        for (std::size_t index = 0; index < material.parameters.size(); ++index)
        {
            const auto& expected = material.parameters[index];
            const auto& actual = reflection.parameters[index];
            if (expected.id != actual.id || expected.type != actual.type || expected.offset != actual.offset ||
                expected.size != actual.size)
                return result::failure("Cooked shader parameter layout differs from material contract");
        }
        program.parameter_block_size = std::max(program.parameter_block_size, reflection.parameter_block_size);
        program.passes.push_back({.pass = binding.pass,
                                  .permutation = binding.permutation,
                                  .compiled = package.compiled,
                                  .virtual_shadow_sampling = matched->artifact_name.ends_with(".forward.vsm")});
    }
    program.parameter_defaults.resize(program.parameter_block_size);
    for (const auto& parameter : program.passes.front().compiled.reflection.parameters)
    {
        if (parameter.default_value.empty()) continue;
        if (parameter.offset > program.parameter_defaults.size() ||
            parameter.default_value.size() > program.parameter_defaults.size() - parameter.offset)
            return result::failure("Cooked shader parameter defaults exceed the reflected block");
        std::copy(parameter.default_value.begin(), parameter.default_value.end(),
                  program.parameter_defaults.begin() + parameter.offset);
    }
    return result::success(std::move(program));
}

core::result<material_descriptor, std::string>
make_cooked_material_descriptor(const cooked_material_metadata& material,
                                std::shared_ptr<const material_runtime_program> program)
{
    using result = core::result<material_descriptor, std::string>;
    if (!program || !material_runtime_program_compatible(*program) || program->passes.empty() ||
        !material.compiled.package.valid())
        return result::failure("Cooked material has no compatible executable runtime program");

    const auto authored = nlohmann::json::parse(material.canonical_document_json, nullptr, false);
    if (authored.is_discarded() || !authored.is_object() || !authored.contains("version") ||
        !authored["version"].is_number_integer() || authored["version"] != 4)
        return result::failure("Cooked material has an invalid authored document version");

    // These are the exact v4 authoring enum spellings. Do not silently
    // reinterpret an unknown value as the standard/opaque default.
    const auto parse_string = [&](std::string_view key, std::string_view fallback) -> std::optional<std::string>
    {
        const auto it = authored.find(std::string(key));
        if (it == authored.end()) return std::string(fallback);
        if (!it->is_string()) return std::nullopt;
        return it->get<std::string>();
    };
    const auto domain = parse_string("domain", "surface");
    const auto shading = parse_string("shadingModel", "standard");
    const auto blending = parse_string("blendMode", "opaque");
    if (!domain || *domain != "surface" || !shading || !blending)
        return result::failure("Cooked material has unsupported surface state");

    material_descriptor descriptor;
    if (*shading == "standard")
        descriptor.shading_model = material_shading_model::standard;
    else if (*shading == "skin")
        descriptor.shading_model = material_shading_model::skin;
    else if (*shading == "transmission")
        descriptor.shading_model = material_shading_model::transmission;
    else if (*shading == "unlit")
        descriptor.shading_model = material_shading_model::unlit;
    else if (*shading == "customLit" || *shading == "custom_lit")
        descriptor.shading_model = material_shading_model::custom_lit;
    else
        return result::failure("Cooked material has an unknown authored shading model");

    if (*blending == "opaque")
        descriptor.alpha_mode = material_alpha_mode::opaque;
    else if (*blending == "masked")
        descriptor.alpha_mode = material_alpha_mode::masked;
    else if (*blending == "blend")
        descriptor.alpha_mode = material_alpha_mode::blend;
    else
        return result::failure("Cooked material has an unknown authored blend mode");

    for (const auto key : {"doubleSided", "castShadows"})
    {
        const auto it = authored.find(key);
        if (it != authored.end() && !it->is_boolean())
            return result::failure(std::string("Cooked material has invalid ") + key + " state");
    }
    descriptor.double_sided = authored.value("doubleSided", false);
    descriptor.cast_shadows = authored.value("castShadows", true);
    descriptor.shader_package = material.compiled.package;
    descriptor.runtime_program = std::move(program);
    descriptor.render_path = material_render_path::clustered_forward;

    const auto* forward = find_material_runtime_pass(*descriptor.runtime_program, material_pass::forward, false);
    if (!forward)
        return result::failure("Cooked material lacks the conventional forward pass required on Android");
    descriptor.shader_permutation = forward->permutation;
    return result::success(std::move(descriptor));
}
} // namespace arc::render
