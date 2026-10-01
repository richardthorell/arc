#include <arc/render/renderer_profile.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace arc::render
{
namespace
{
using json = nlohmann::json;
constexpr std::uint64_t mebibyte = 1024ull * 1024ull;
constexpr std::uint64_t gibibyte = 1024ull * mebibyte;

struct parse_failure final : std::runtime_error
{
    parse_failure(std::string field_value, std::string message_value)
        : std::runtime_error(message_value), field(std::move(field_value))
    {
    }

    std::string field;
};

template <typename Value>
void read_unsigned(const json& source, std::string_view key, std::optional<Value>& destination, std::string_view field,
                   Value maximum = std::numeric_limits<Value>::max())
{
    const auto iterator = source.find(key);
    if (iterator == source.end()) return;
    if (!iterator->is_number_unsigned() && !iterator->is_number_integer())
        throw parse_failure(std::string(field), "must be a non-negative integer");
    const auto value = iterator->get<std::int64_t>();
    if (value < 0 || static_cast<std::uint64_t>(value) > static_cast<std::uint64_t>(maximum))
        throw parse_failure(std::string(field), "is outside the supported range");
    destination = static_cast<Value>(value);
}

void read_bytes(const json& source, std::string_view bytes_key, std::string_view mebibytes_key,
                std::optional<std::uint64_t>& destination, std::string_view field)
{
    read_unsigned(source, bytes_key, destination, field);
    std::optional<std::uint64_t> mebibytes;
    read_unsigned(source, mebibytes_key, mebibytes, field);
    if (!mebibytes) return;
    if (*mebibytes > std::numeric_limits<std::uint64_t>::max() / mebibyte)
        throw parse_failure(std::string(field), "is too large");
    destination = *mebibytes * mebibyte;
}

void read_float(const json& source, std::string_view key, std::optional<float>& destination, std::string_view field,
                float minimum, float maximum)
{
    const auto iterator = source.find(key);
    if (iterator == source.end()) return;
    if (!iterator->is_number()) throw parse_failure(std::string(field), "must be numeric");
    const auto value = iterator->get<float>();
    if (!std::isfinite(value) || value < minimum || value > maximum)
        throw parse_failure(std::string(field), "is outside the supported range");
    destination = value;
}

void read_bool(const json& source, std::string_view key, std::optional<bool>& destination, std::string_view field)
{
    const auto iterator = source.find(key);
    if (iterator == source.end()) return;
    if (!iterator->is_boolean()) throw parse_failure(std::string(field), "must be a boolean");
    destination = iterator->get<bool>();
}

std::string string_value(const json& source, std::string_view key, std::string_view field)
{
    const auto iterator = source.find(key);
    if (iterator == source.end()) return {};
    if (!iterator->is_string()) throw parse_failure(std::string(field), "must be a string");
    return iterator->get<std::string>();
}

render_quality_tier parse_quality(std::string_view value, std::string_view field)
{
    if (value == "auto") return render_quality_tier::auto_select;
    if (value == "low") return render_quality_tier::low;
    if (value == "standard" || value == "medium") return render_quality_tier::medium;
    if (value == "high") return render_quality_tier::high;
    if (value == "ultra") return render_quality_tier::ultra;
    throw parse_failure(std::string(field), "must be auto, low, standard, high, or ultra");
}

render_path parse_path(std::string_view value, std::string_view field)
{
    if (value == "auto") return render_path::auto_select;
    if (value == "forward" || value == "forwardPlus") return render_path::forward_plus;
    if (value == "deferred") return render_path::deferred;
    throw parse_failure(std::string(field), "must be auto, forwardPlus, or deferred");
}

anti_aliasing_method parse_anti_aliasing(std::string_view value, std::string_view field)
{
    if (value == "auto") return anti_aliasing_method::auto_select;
    if (value == "disabled") return anti_aliasing_method::disabled;
    if (value == "fxaa") return anti_aliasing_method::fxaa;
    if (value == "taa") return anti_aliasing_method::taa;
    if (value == "taau") return anti_aliasing_method::taau;
    throw parse_failure(std::string(field), "must be auto, disabled, fxaa, taa, or taau");
}

render_scalability_tier parse_scalability_tier(std::string_view value, std::string_view field)
{
    if (value == "auto") return render_scalability_tier::automatic;
    if (value == "constrained") return render_scalability_tier::constrained;
    if (value == "balanced") return render_scalability_tier::balanced;
    if (value == "performance") return render_scalability_tier::performance;
    if (value == "premium") return render_scalability_tier::premium;
    throw parse_failure(std::string(field), "must be auto, constrained, balanced, performance, or premium");
}

framework::device_form_factor parse_form_factor(std::string_view value, std::string_view field)
{
    if (value == "desktop") return framework::device_form_factor::desktop;
    if (value == "handheld") return framework::device_form_factor::handheld;
    if (value == "tablet") return framework::device_form_factor::tablet;
    if (value == "console") return framework::device_form_factor::console;
    if (value == "server") return framework::device_form_factor::server;
    if (value == "unknown") return framework::device_form_factor::unknown;
    throw parse_failure(std::string(field), "contains an unknown form factor");
}

const json& object_or_empty(const json& source, std::string_view key, std::string_view field)
{
    static const json empty = json::object();
    const auto iterator = source.find(key);
    if (iterator == source.end()) return empty;
    if (!iterator->is_object()) throw parse_failure(std::string(field), "must be an object");
    return *iterator;
}

void parse_settings(const json& source, renderer_profile_overrides& result, std::string_view prefix)
{
    if (!source.is_object()) throw parse_failure(std::string(prefix), "must be an object");
    const auto field = [&](std::string_view name) { return std::string(prefix) + std::string(name); };

    if (const auto value = string_value(source, "quality", field("quality")); !value.empty())
        result.quality = parse_quality(value, field("quality"));
    if (const auto value = string_value(source, "qualityTier", field("qualityTier")); !value.empty())
        result.quality = parse_quality(value, field("qualityTier"));
    if (const auto value = string_value(source, "path", field("path")); !value.empty())
        result.path = parse_path(value, field("path"));
    if (const auto value = string_value(source, "antiAliasing", field("antiAliasing")); !value.empty())
        result.anti_aliasing = parse_anti_aliasing(value, field("antiAliasing"));
    read_bool(source, "dynamicResolution", result.dynamic_resolution, field("dynamicResolution"));
    read_float(source, "targetFrameTimeMs", result.target_frame_time_ms, field("targetFrameTimeMs"), 1.0f, 1000.0f);
    read_float(source, "minimumRenderScale", result.minimum_render_scale, field("minimumRenderScale"), 0.25f, 1.0f);
    read_float(source, "maximumRenderScale", result.maximum_render_scale, field("maximumRenderScale"), 0.25f, 2.0f);

    const auto& tiers = object_or_empty(source, "tiers", field("tiers"));
    if (const auto value = string_value(tiers, "cpu", field("tiers.cpu")); !value.empty())
        result.cpu_tier = parse_scalability_tier(value, field("tiers.cpu"));
    if (const auto value = string_value(tiers, "gpu", field("tiers.gpu")); !value.empty())
        result.gpu_tier = parse_scalability_tier(value, field("tiers.gpu"));
    if (const auto value = string_value(tiers, "memory", field("tiers.memory")); !value.empty())
        result.memory_tier = parse_scalability_tier(value, field("tiers.memory"));

    const auto& geometry = object_or_empty(source, "virtualGeometry", field("virtualGeometry"));
    read_float(geometry, "projectedError", result.geometry_error_threshold, field("virtualGeometry.projectedError"),
               0.01f, 64.0f);
    read_float(geometry, "minimumProjectedError", result.minimum_geometry_error_threshold,
               field("virtualGeometry.minimumProjectedError"), 0.01f, 64.0f);
    read_float(geometry, "maximumProjectedError", result.maximum_geometry_error_threshold,
               field("virtualGeometry.maximumProjectedError"), 0.01f, 64.0f);
    read_bytes(geometry, "gpuCacheBytes", "gpuCacheMiB", result.virtual_geometry_gpu_budget_bytes,
               field("virtualGeometry.gpuCache"));
    read_bytes(geometry, "cpuCacheBytes", "cpuCacheMiB", result.virtual_geometry_cpu_budget_bytes,
               field("virtualGeometry.cpuCache"));
    read_unsigned(geometry, "requestLimit", result.virtual_geometry_request_limit,
                  field("virtualGeometry.requestLimit"));
    read_float(geometry, "computeCrossoverPixels", result.virtual_geometry_compute_crossover_pixels,
               field("virtualGeometry.computeCrossoverPixels"), 0.0f, 65536.0f);
    read_float(geometry, "hardwareCrossoverPixels", result.virtual_geometry_hardware_crossover_pixels,
               field("virtualGeometry.hardwareCrossoverPixels"), 0.0f, 65536.0f);

    const auto& textures = object_or_empty(source, "textureStreaming", field("textureStreaming"));
    read_bytes(textures, "gpuBudgetBytes", "gpuBudgetMiB", result.texture_gpu_budget_bytes,
               field("textureStreaming.gpuBudget"));
    read_bytes(textures, "cpuBudgetBytes", "cpuBudgetMiB", result.texture_cpu_budget_bytes,
               field("textureStreaming.cpuBudget"));
    read_bytes(textures, "uploadBudgetBytes", "uploadBudgetMiB", result.texture_upload_budget_per_frame,
               field("textureStreaming.uploadBudget"));
    read_unsigned(textures, "requestLimit", result.texture_request_limit, field("textureStreaming.requestLimit"));

    const auto& virtual_textures = object_or_empty(source, "virtualTexturing", field("virtualTexturing"));
    read_bytes(virtual_textures, "physicalCacheBytes", "physicalCacheMiB", result.virtual_texture_cache_budget_bytes,
               field("virtualTexturing.physicalCache"));

    const auto& shadows = object_or_empty(source, "shadows", field("shadows"));
    read_unsigned(shadows, "directionalCascades", result.directional_shadow_cascades,
                  field("shadows.directionalCascades"), 8u);
    read_unsigned(shadows, "directionalResolution", result.directional_shadow_resolution,
                  field("shadows.directionalResolution"), 16384u);
    read_unsigned(shadows, "localAtlasResolution", result.local_shadow_atlas_resolution,
                  field("shadows.localAtlasResolution"), 32768u);
    read_float(shadows, "minimumScale", result.minimum_shadow_resolution_scale, field("shadows.minimumScale"), 0.25f,
               1.0f);
    read_float(shadows, "maximumScale", result.maximum_shadow_resolution_scale, field("shadows.maximumScale"), 0.25f,
               1.0f);

    const auto& volumetrics = object_or_empty(source, "volumetrics", field("volumetrics"));
    read_float(volumetrics, "minimumScale", result.minimum_volumetric_resolution_scale,
               field("volumetrics.minimumScale"), 0.25f, 1.0f);
    read_float(volumetrics, "maximumScale", result.maximum_volumetric_resolution_scale,
               field("volumetrics.maximumScale"), 0.25f, 1.0f);

    const auto& terrain = object_or_empty(source, "terrain", field("terrain"));
    read_float(terrain, "geometryErrorScale", result.terrain_geometry_error_scale, field("terrain.geometryErrorScale"),
               0.1f, 8.0f);
    const auto& post = object_or_empty(source, "postProcessing", field("postProcessing"));
    read_float(post, "quality", result.post_process_quality, field("postProcessing.quality"), 0.0f, 1.0f);
}

void parse_flat_project_settings(const json& source, renderer_profile_overrides& result)
{
    const auto read_string_setting = [&](std::string_view key) -> std::string
    {
        const auto iterator = source.find(key);
        if (iterator == source.end()) return {};
        if (!iterator->is_string()) throw parse_failure(std::string(key), "must be a string");
        return iterator->get<std::string>();
    };
    if (const auto value = read_string_setting("renderer.qualityTier"); !value.empty())
        result.quality = parse_quality(value, "renderer.qualityTier");
    if (const auto value = read_string_setting("renderer.antiAliasing"); !value.empty())
        result.anti_aliasing = parse_anti_aliasing(value, "renderer.antiAliasing");
}

renderer_device_profile_match parse_match(const json& source, std::string_view prefix)
{
    if (!source.is_object()) throw parse_failure(std::string(prefix), "must be an object");
    renderer_device_profile_match result;
    const auto field = [&](std::string_view name) { return std::string(prefix) + std::string(name); };
    if (const auto value = string_value(source, "gpuClass", field("gpuClass")); !value.empty())
    {
        if (value == "any")
            result.gpu_class = renderer_gpu_class::any;
        else if (value == "integrated")
            result.gpu_class = renderer_gpu_class::integrated;
        else if (value == "discrete")
            result.gpu_class = renderer_gpu_class::discrete;
        else
            throw parse_failure(field("gpuClass"), "must be any, integrated, or discrete");
    }
    if (const auto value = string_value(source, "formFactor", field("formFactor")); !value.empty())
        result.form_factor = parse_form_factor(value, field("formFactor"));

    std::optional<std::uint32_t> processors;
    read_unsigned(source, "minimumLogicalProcessors", processors, field("minimumLogicalProcessors"));
    result.minimum_logical_processors = processors.value_or(0u);
    processors.reset();
    read_unsigned(source, "maximumLogicalProcessors", processors, field("maximumLogicalProcessors"));
    result.maximum_logical_processors = processors.value_or(0u);

    std::optional<std::uint64_t> bytes;
    read_bytes(source, "minimumSystemMemoryBytes", "minimumSystemMemoryMiB", bytes, field("minimumSystemMemory"));
    result.minimum_system_memory_bytes = bytes.value_or(0u);
    bytes.reset();
    read_bytes(source, "maximumSystemMemoryBytes", "maximumSystemMemoryMiB", bytes, field("maximumSystemMemory"));
    result.maximum_system_memory_bytes = bytes.value_or(0u);
    bytes.reset();
    read_bytes(source, "minimumGpuMemoryBytes", "minimumGpuMemoryMiB", bytes, field("minimumGpuMemory"));
    result.minimum_gpu_memory_bytes = bytes.value_or(0u);
    bytes.reset();
    read_bytes(source, "maximumGpuMemoryBytes", "maximumGpuMemoryMiB", bytes, field("maximumGpuMemory"));
    result.maximum_gpu_memory_bytes = bytes.value_or(0u);

    if (const auto iterator = source.find("requires"); iterator != source.end())
    {
        if (!iterator->is_array()) throw parse_failure(field("requires"), "must be an array");
        static const std::unordered_set<std::string> known{"computeShaders",
                                                           "storageBuffers",
                                                           "storageImages",
                                                           "descriptorIndexing",
                                                           "virtualGeometryCompute",
                                                           "virtualGeometryIndexed",
                                                           "virtualGeometryMeshShader",
                                                           "meshShaders",
                                                           "rayTracing",
                                                           "sparseResources",
                                                           "virtualTextures"};
        for (std::size_t index = 0; index < iterator->size(); ++index)
        {
            if (!(*iterator)[index].is_string())
                throw parse_failure(field("requires[" + std::to_string(index) + "]"), "must be a string");
            auto value = (*iterator)[index].get<std::string>();
            if (!known.contains(value))
                throw parse_failure(field("requires[" + std::to_string(index) + "]"),
                                    "contains an unknown renderer feature");
            result.required_features.push_back(std::move(value));
        }
    }
    return result;
}

bool feature_available(std::string_view feature, const render_capabilities& capabilities) noexcept
{
    if (feature == "computeShaders") return capabilities.compute_shaders;
    if (feature == "storageBuffers") return capabilities.storage_buffers;
    if (feature == "storageImages") return capabilities.storage_images;
    if (feature == "descriptorIndexing") return capabilities.descriptor_indexing;
    if (feature == "virtualGeometryCompute") return capabilities.virtual_geometry_compute;
    if (feature == "virtualGeometryIndexed") return capabilities.virtual_geometry_indexed;
    if (feature == "virtualGeometryMeshShader") return capabilities.virtual_geometry_mesh_shader;
    if (feature == "meshShaders") return capabilities.mesh_shaders;
    if (feature == "rayTracing") return capabilities.ray_tracing;
    if (feature == "sparseResources") return capabilities.sparse_resources;
    if (feature == "virtualTextures")
        return capabilities.virtual_texture_feedback && capabilities.virtual_texture_sampling;
    return false;
}

std::uint64_t available_gpu_memory(const render_capabilities& capabilities) noexcept
{
    if (capabilities.memory_budget != 0) return capabilities.memory_budget;
    if (capabilities.dedicated_video_memory != 0) return capabilities.dedicated_video_memory;
    return capabilities.shared_system_memory;
}

bool matches(const renderer_device_profile_match& match, const render_capabilities& capabilities,
             const framework::platform_capabilities& platform) noexcept
{
    if (match.gpu_class == renderer_gpu_class::integrated && !capabilities.integrated_gpu) return false;
    if (match.gpu_class == renderer_gpu_class::discrete && !capabilities.discrete_gpu) return false;
    if (match.form_factor && *match.form_factor != platform.form_factor) return false;
    if (match.minimum_logical_processors != 0 && platform.logical_processor_count < match.minimum_logical_processors)
        return false;
    if (match.maximum_logical_processors != 0 && platform.logical_processor_count > match.maximum_logical_processors)
        return false;
    if (match.minimum_system_memory_bytes != 0 && platform.system_memory_bytes < match.minimum_system_memory_bytes)
        return false;
    if (match.maximum_system_memory_bytes != 0 && platform.system_memory_bytes > match.maximum_system_memory_bytes)
        return false;
    const auto gpu_memory = available_gpu_memory(capabilities);
    if (match.minimum_gpu_memory_bytes != 0 && gpu_memory < match.minimum_gpu_memory_bytes) return false;
    if (match.maximum_gpu_memory_bytes != 0 && gpu_memory > match.maximum_gpu_memory_bytes) return false;
    return std::all_of(match.required_features.begin(), match.required_features.end(),
                       [&](const std::string& feature) { return feature_available(feature, capabilities); });
}

std::uint32_t specificity(const renderer_device_profile_match& match) noexcept
{
    return (match.gpu_class != renderer_gpu_class::any ? 1u : 0u) + (match.form_factor ? 1u : 0u) +
           (match.minimum_logical_processors != 0 ? 1u : 0u) + (match.maximum_logical_processors != 0 ? 1u : 0u) +
           (match.minimum_system_memory_bytes != 0 ? 1u : 0u) + (match.maximum_system_memory_bytes != 0 ? 1u : 0u) +
           (match.minimum_gpu_memory_bytes != 0 ? 1u : 0u) + (match.maximum_gpu_memory_bytes != 0 ? 1u : 0u) +
           static_cast<std::uint32_t>(match.required_features.size());
}

render_scalability_tier cpu_tier(const framework::platform_capabilities& platform) noexcept
{
    if ((platform.logical_processor_count != 0 && platform.logical_processor_count <= 4u) ||
        (platform.system_memory_bytes != 0 && platform.system_memory_bytes < 8ull * gibibyte))
        return render_scalability_tier::constrained;
    if (platform.logical_processor_count >= 16u) return render_scalability_tier::performance;
    return render_scalability_tier::balanced;
}

render_scalability_tier gpu_tier(const render_capabilities& capabilities) noexcept
{
    const auto memory = available_gpu_memory(capabilities);
    if (capabilities.integrated_gpu || (memory != 0 && memory < 2ull * gibibyte))
        return render_scalability_tier::constrained;
    if (memory >= 12ull * gibibyte) return render_scalability_tier::premium;
    if (memory >= 6ull * gibibyte) return render_scalability_tier::performance;
    return render_scalability_tier::balanced;
}

render_scalability_tier memory_tier(const render_capabilities& capabilities,
                                    const framework::platform_capabilities& platform) noexcept
{
    const auto gpu_memory = available_gpu_memory(capabilities);
    if ((platform.system_memory_bytes != 0 && platform.system_memory_bytes < 8ull * gibibyte) ||
        (gpu_memory != 0 && gpu_memory < 2ull * gibibyte))
        return render_scalability_tier::constrained;
    if (platform.system_memory_bytes >= 32ull * gibibyte && gpu_memory >= 12ull * gibibyte)
        return render_scalability_tier::premium;
    if (platform.system_memory_bytes >= 16ull * gibibyte && gpu_memory >= 6ull * gibibyte)
        return render_scalability_tier::performance;
    return render_scalability_tier::balanced;
}

void apply_settings(render_quality_profile& profile, const renderer_profile_overrides& overrides)
{
#define ARC_APPLY_PROFILE(member)                                                                                      \
    if (overrides.member) profile.member = *overrides.member
    ARC_APPLY_PROFILE(target_frame_time_ms);
    ARC_APPLY_PROFILE(minimum_render_scale);
    ARC_APPLY_PROFILE(maximum_render_scale);
    ARC_APPLY_PROFILE(geometry_error_threshold);
    ARC_APPLY_PROFILE(minimum_geometry_error_threshold);
    ARC_APPLY_PROFILE(maximum_geometry_error_threshold);
    ARC_APPLY_PROFILE(directional_shadow_cascades);
    ARC_APPLY_PROFILE(directional_shadow_resolution);
    ARC_APPLY_PROFILE(local_shadow_atlas_resolution);
    ARC_APPLY_PROFILE(minimum_shadow_resolution_scale);
    ARC_APPLY_PROFILE(maximum_shadow_resolution_scale);
    ARC_APPLY_PROFILE(minimum_volumetric_resolution_scale);
    ARC_APPLY_PROFILE(maximum_volumetric_resolution_scale);
    ARC_APPLY_PROFILE(virtual_geometry_gpu_budget_bytes);
    ARC_APPLY_PROFILE(virtual_geometry_cpu_budget_bytes);
    ARC_APPLY_PROFILE(virtual_geometry_request_limit);
    ARC_APPLY_PROFILE(virtual_geometry_compute_crossover_pixels);
    ARC_APPLY_PROFILE(virtual_geometry_hardware_crossover_pixels);
    ARC_APPLY_PROFILE(texture_gpu_budget_bytes);
    ARC_APPLY_PROFILE(texture_cpu_budget_bytes);
    ARC_APPLY_PROFILE(texture_upload_budget_per_frame);
    ARC_APPLY_PROFILE(texture_request_limit);
    ARC_APPLY_PROFILE(virtual_texture_cache_budget_bytes);
    ARC_APPLY_PROFILE(terrain_geometry_error_scale);
    ARC_APPLY_PROFILE(post_process_quality);
#undef ARC_APPLY_PROFILE
}

void normalize(render_quality_profile& profile) noexcept
{
    profile.minimum_render_scale = std::clamp(profile.minimum_render_scale, 0.25f, 1.0f);
    profile.maximum_render_scale = std::clamp(profile.maximum_render_scale, profile.minimum_render_scale, 2.0f);
    profile.minimum_geometry_error_threshold = std::max(0.01f, profile.minimum_geometry_error_threshold);
    profile.maximum_geometry_error_threshold =
        std::max(profile.minimum_geometry_error_threshold, profile.maximum_geometry_error_threshold);
    profile.geometry_error_threshold =
        std::clamp(profile.geometry_error_threshold, profile.minimum_geometry_error_threshold,
                   profile.maximum_geometry_error_threshold);
    profile.minimum_shadow_resolution_scale = std::clamp(profile.minimum_shadow_resolution_scale, 0.25f, 1.0f);
    profile.maximum_shadow_resolution_scale =
        std::clamp(profile.maximum_shadow_resolution_scale, profile.minimum_shadow_resolution_scale, 1.0f);
    profile.minimum_volumetric_resolution_scale = std::clamp(profile.minimum_volumetric_resolution_scale, 0.25f, 1.0f);
    profile.maximum_volumetric_resolution_scale =
        std::clamp(profile.maximum_volumetric_resolution_scale, profile.minimum_volumetric_resolution_scale, 1.0f);
    profile.target_frame_time_ms = std::max(1.0f, profile.target_frame_time_ms);
    profile.virtual_geometry_request_limit = std::max(1u, profile.virtual_geometry_request_limit);
    profile.texture_request_limit = std::max(1u, profile.texture_request_limit);
    profile.virtual_geometry_hardware_crossover_pixels =
        std::max(profile.virtual_geometry_compute_crossover_pixels, profile.virtual_geometry_hardware_crossover_pixels);
    profile.terrain_geometry_error_scale = std::clamp(profile.terrain_geometry_error_scale, 0.1f, 8.0f);
    profile.post_process_quality = std::clamp(profile.post_process_quality, 0.0f, 1.0f);
}

} // namespace

renderer_profile_document_result parse_renderer_profile_document(std::string_view source)
{
    renderer_profile_document_result result;
    if (source.empty()) return result;
    try
    {
        const auto root = json::parse(source);
        if (!root.is_object()) throw parse_failure("", "renderer profile root must be an object");
        if (const auto format = string_value(root, "format", "format");
            !format.empty() && format != renderer_profile_format)
            throw parse_failure("format", "is not an ARC renderer profile document");
        if (const auto iterator = root.find("formatVersion"); iterator != root.end())
        {
            if (!iterator->is_number_unsigned() && !iterator->is_number_integer())
                throw parse_failure("formatVersion", "must be an integer");
            if (iterator->get<std::uint32_t>() != renderer_profile_format_version)
                throw parse_failure("formatVersion", "is not supported");
        }
        result.document.preferred_profile_id = string_value(root, "preferredProfile", "preferredProfile");
        if (const auto iterator = root.find("deviceProfiles"); iterator != root.end())
        {
            if (!iterator->is_array()) throw parse_failure("deviceProfiles", "must be an array");
            std::unordered_set<std::string> ids;
            for (std::size_t index = 0; index < iterator->size(); ++index)
            {
                const auto& value = (*iterator)[index];
                const auto prefix = "deviceProfiles[" + std::to_string(index) + "].";
                if (!value.is_object()) throw parse_failure(prefix, "must be an object");
                renderer_device_profile profile;
                profile.id = string_value(value, "id", prefix + "id");
                if (profile.id.empty()) throw parse_failure(prefix + "id", "is required");
                if (!ids.emplace(profile.id).second) throw parse_failure(prefix + "id", "must be unique");
                if (const auto priority = value.find("priority"); priority != value.end())
                {
                    if (!priority->is_number_integer()) throw parse_failure(prefix + "priority", "must be an integer");
                    profile.priority = priority->get<std::int32_t>();
                }
                profile.match = parse_match(object_or_empty(value, "match", prefix + "match"), prefix + "match.");
                parse_settings(object_or_empty(value, "settings", prefix + "settings"), profile.settings,
                               prefix + "settings.");
                result.document.device_profiles.push_back(std::move(profile));
            }
        }
        if (const auto iterator = root.find("overrides"); iterator != root.end())
            parse_settings(*iterator, result.document.project_overrides, "overrides.");
        parse_flat_project_settings(root, result.document.project_overrides);
    }
    catch (const parse_failure& failure)
    {
        result.error = renderer_profile_error{.field = failure.field, .message = failure.what()};
    }
    catch (const std::exception& exception)
    {
        result.error = renderer_profile_error{.message = exception.what()};
    }
    return result;
}

renderer_profile_document_result load_renderer_profile_document(const std::filesystem::path& path)
{
    std::error_code error;
    if (!std::filesystem::exists(path, error))
    {
        if (!error) return {};
        return {.error = renderer_profile_error{.path = path, .message = error.message()}};
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {.error = renderer_profile_error{.path = path, .message = "could not open renderer profile"}};
    const std::string source{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    auto result = parse_renderer_profile_document(source);
    if (result.error) result.error->path = path;
    return result;
}

renderer_profile_resolution resolve_renderer_profile(const renderer_profile_document& document,
                                                     const renderer_profile_overrides& runtime_overrides,
                                                     const render_capabilities& capabilities,
                                                     const framework::platform_capabilities& platform,
                                                     std::string_view requested_profile_id)
{
    renderer_profile_resolution result;
    result.cpu_tier = cpu_tier(platform);
    result.gpu_tier = gpu_tier(capabilities);
    result.memory_tier = memory_tier(capabilities, platform);

    const renderer_device_profile* selected = nullptr;
    const auto preferred =
        requested_profile_id.empty() ? std::string_view(document.preferred_profile_id) : requested_profile_id;
    if (!preferred.empty())
    {
        const auto iterator =
            std::find_if(document.device_profiles.begin(), document.device_profiles.end(),
                         [&](const renderer_device_profile& profile) { return profile.id == preferred; });
        if (iterator == document.device_profiles.end())
            result.diagnostics.push_back("requested renderer profile '" + std::string(preferred) + "' was not found");
        else if (!matches(iterator->match, capabilities, platform))
            result.diagnostics.push_back("requested renderer profile '" + std::string(preferred) +
                                         "' does not match the active device capabilities");
        else
            selected = &*iterator;
    }
    if (!selected)
    {
        for (const auto& candidate : document.device_profiles)
        {
            if (!matches(candidate.match, capabilities, platform)) continue;
            if (!selected || candidate.priority > selected->priority ||
                (candidate.priority == selected->priority &&
                 specificity(candidate.match) > specificity(selected->match)) ||
                (candidate.priority == selected->priority &&
                 specificity(candidate.match) == specificity(selected->match) && candidate.id < selected->id))
                selected = &candidate;
        }
    }

    const auto* device_settings = selected ? &selected->settings : nullptr;
    if (selected) result.device_profile_id = selected->id;
    const auto last_quality = [&]() -> std::optional<render_quality_tier>
    {
        if (runtime_overrides.quality) return runtime_overrides.quality;
        if (document.project_overrides.quality) return document.project_overrides.quality;
        if (device_settings && device_settings->quality) return device_settings->quality;
        return std::nullopt;
    }();
    result.requested_quality = last_quality.value_or(render_quality_tier::auto_select);
    const bool constrained = result.cpu_tier == render_scalability_tier::constrained ||
                             result.gpu_tier == render_scalability_tier::constrained ||
                             result.memory_tier == render_scalability_tier::constrained;
    result.quality = result.requested_quality == render_quality_tier::auto_select
                         ? (constrained ? render_quality_tier::low : render_quality_tier::medium)
                         : result.requested_quality;
    const auto gpu_memory = available_gpu_memory(capabilities);
    if (result.quality == render_quality_tier::ultra && gpu_memory != 0 && gpu_memory < 12ull * gibibyte)
    {
        result.quality = render_quality_tier::high;
        result.diagnostics.push_back(
            "ultra quality requires at least 12 GiB of available GPU memory; using high limits");
    }
    if (result.quality == render_quality_tier::high && gpu_memory != 0 && gpu_memory < 6ull * gibibyte)
    {
        result.quality = render_quality_tier::medium;
        result.diagnostics.push_back(
            "high quality requires at least 6 GiB of available GPU memory; using standard limits");
    }
    if (result.requested_quality == render_quality_tier::auto_select)
        result.diagnostics.push_back(constrained ? "auto-selected low quality from device capability tiers"
                                                 : "auto-selected standard quality from device capability tiers");

    result.profile = quality_profile(result.quality);
    if (device_settings) apply_settings(result.profile, *device_settings);
    apply_settings(result.profile, document.project_overrides);
    apply_settings(result.profile, runtime_overrides);
    result.profile.quality = result.quality;
    normalize(result.profile);

    const auto apply_metadata = [&](const renderer_profile_overrides& overrides)
    {
        if (overrides.path) result.path = *overrides.path;
        if (overrides.anti_aliasing) result.anti_aliasing = *overrides.anti_aliasing;
        if (overrides.cpu_tier) result.cpu_tier = *overrides.cpu_tier;
        if (overrides.gpu_tier) result.gpu_tier = *overrides.gpu_tier;
        if (overrides.memory_tier) result.memory_tier = *overrides.memory_tier;
        if (overrides.dynamic_resolution) result.dynamic_resolution = *overrides.dynamic_resolution;
    };
    if (device_settings) apply_metadata(*device_settings);
    apply_metadata(document.project_overrides);
    apply_metadata(runtime_overrides);
    return result;
}

} // namespace arc::render
