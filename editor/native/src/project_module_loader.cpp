#include "project_module_loader.h"

#include <arc/diagnostics/diagnostics.h>

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <system_error>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace arc::editor
{
namespace
{
void module_log(void*, const char* category, const char* message)
{
    diagnostics::info(category ? category : "project.module", message ? message : "");
}

bool stable_component_id(std::string_view value)
{
    return value.size() == 32 && std::all_of(value.begin(), value.end(), [](unsigned char character)
                                             { return std::isxdigit(character) != 0; });
}

std::vector<project_component_schema> copy_components(const project::game_module_descriptor_v1& descriptor,
                                                      std::string& error)
{
    std::vector<project_component_schema> result;
    result.reserve(descriptor.component_count);
    for (std::size_t component_index = 0; component_index < descriptor.component_count; ++component_index)
    {
        const auto& source = descriptor.components[component_index];
        if (!source.stable_id || !stable_component_id(source.stable_id) || !source.canonical_name ||
            !source.display_name || source.schema_version == 0 || (source.field_count && !source.fields))
        {
            error = "project module contains an invalid component schema";
            return {};
        }
        if (std::any_of(result.begin(), result.end(), [&](const auto& current)
                        { return current.stable_id == source.stable_id; }))
        {
            error = "project module contains duplicate component stable IDs";
            return {};
        }
        project_component_schema component{.stable_id = source.stable_id,
                                           .canonical_name = source.canonical_name,
                                           .display_name = source.display_name,
                                           .category = source.category ? source.category : "Project",
                                           .tooltip = source.tooltip ? source.tooltip : "",
                                           .schema_version = source.schema_version};
        component.fields.reserve(source.field_count);
        for (std::size_t field_index = 0; field_index < source.field_count; ++field_index)
        {
            const auto& field = source.fields[field_index];
            if (!field.stable_id || !field.name || !field.display_name || !field.default_json ||
                std::any_of(component.fields.begin(), component.fields.end(), [&](const auto& current)
                            { return current.stable_id == field.stable_id; }))
            {
                error = "project module contains an invalid or duplicate field stable ID";
                return {};
            }
            component.fields.push_back({.stable_id = field.stable_id,
                                        .name = field.name,
                                        .display_name = field.display_name,
                                        .category = field.category ? field.category : "",
                                        .tooltip = field.tooltip ? field.tooltip : "",
                                        .kind = field.kind,
                                        .flags = field.flags,
                                        .default_json = field.default_json,
                                        .minimum = field.minimum,
                                        .maximum = field.maximum,
                                        .has_minimum = field.has_minimum,
                                        .has_maximum = field.has_maximum,
                                        .asset_type_restriction = field.asset_type_restriction
                                                                      ? field.asset_type_restriction
                                                                      : "",
                                        .entity_component_restriction = field.entity_component_restriction
                                                                           ? field.entity_component_restriction
                                                                           : ""});
        }
        result.push_back(std::move(component));
    }
    return result;
}

std::vector<project_registration_schema> copy_registrations(const project::game_module_descriptor_v1& descriptor,
                                                            std::string& error)
{
    if (descriptor.registration_count && !descriptor.registrations)
    {
        error = "project module registration array is missing";
        return {};
    }
    std::vector<project_registration_schema> result;
    result.reserve(descriptor.registration_count);
    for (std::size_t index = 0; index < descriptor.registration_count; ++index)
    {
        const auto& source = descriptor.registrations[index];
        if (!source.stable_id || !*source.stable_id || !source.name || !*source.name ||
            std::any_of(result.begin(), result.end(), [&](const auto& current)
                        { return current.stable_id == source.stable_id; }))
        {
            error = "project module contains an invalid or duplicate registration ID";
            return {};
        }
        result.push_back({source.kind, source.stable_id, source.name});
    }
    return result;
}

module_reload_classification classify(const std::vector<project_component_schema>& previous,
                                      const std::vector<project_component_schema>& next)
{
    if (previous.empty()) return module_reload_classification::initial_load;
    for (const auto& old_component : previous)
    {
        const auto component = std::find_if(next.begin(), next.end(), [&](const auto& candidate)
                                            { return candidate.stable_id == old_component.stable_id; });
        if (component == next.end() || component->schema_version < old_component.schema_version)
            return module_reload_classification::native_host_restart_required;
        for (const auto& old_field : old_component.fields)
        {
            const auto field = std::find_if(component->fields.begin(), component->fields.end(), [&](const auto& candidate)
                                            { return candidate.stable_id == old_field.stable_id; });
            if (field != component->fields.end() && field->kind != old_field.kind)
                return module_reload_classification::play_session_restart_required;
        }
    }
    return module_reload_classification::safe_hot_reload;
}

const char* classification_message(module_reload_classification value)
{
    switch (value)
    {
    case module_reload_classification::initial_load: return "Project module loaded";
    case module_reload_classification::safe_hot_reload: return "Project module hot reloaded";
    case module_reload_classification::play_session_restart_required: return "Module loaded; play session restart required";
    case module_reload_classification::native_host_restart_required: return "Module schema requires a native host restart";
    }
    return "Project module loaded";
}
} // namespace

project_module_loader::~project_module_loader()
{
    unload();
}

module_reload_result project_module_loader::load(const std::filesystem::path& path, std::string_view engine_version,
                                                 std::string_view project_guid, std::string_view module_id)
{
    if (loaded()) unload();
    return load_generation(path, engine_version, project_guid, module_id, false);
}

module_reload_result project_module_loader::reload(const std::filesystem::path& path, std::string_view engine_version,
                                                   std::string_view project_guid, std::string_view module_id)
{
    return load_generation(path, engine_version, project_guid, module_id, true);
}

module_reload_result project_module_loader::load_generation(const std::filesystem::path& source_path,
                                                            std::string_view engine_version,
                                                            std::string_view project_guid,
                                                            std::string_view module_id, bool /*is_reload*/)
{
    const auto next_generation = generation_ + 1;
    const auto directory = source_path.parent_path() / "HotReload";
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory, filesystem_error);
    std::ostringstream generation_name;
    generation_name << source_path.stem().string() << '_' << std::setfill('0') << std::setw(4) << next_generation
                    << source_path.extension().string();
    auto staged_path = directory / generation_name.str();
    std::filesystem::copy_file(source_path, staged_path, std::filesystem::copy_options::overwrite_existing,
                               filesystem_error);
    if (filesystem_error)
        return {.message = "Could not stage project module generation: " + filesystem_error.message()};

    void* candidate_handle{};
    project::query_game_module_v1 query{};
#if defined(_WIN32)
    HMODULE library = LoadLibraryExW(staged_path.c_str(), nullptr,
                                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    candidate_handle = library;
    if (library) query = reinterpret_cast<project::query_game_module_v1>(GetProcAddress(library, "arc_query_game_module_v1"));
#else
    candidate_handle = dlopen(staged_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (candidate_handle)
        query = reinterpret_cast<project::query_game_module_v1>(dlsym(candidate_handle, "arc_query_game_module_v1"));
#endif
    const auto close_candidate = [&]
    {
        if (candidate_handle)
        {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(candidate_handle));
#else
            dlclose(candidate_handle);
#endif
            candidate_handle = nullptr;
        }
        std::error_code remove_error;
        std::filesystem::remove(staged_path, remove_error);
    };
    if (!candidate_handle || !query)
    {
        close_candidate();
        return {.message = "Project editor module does not export the ARC game-module ABI"};
    }
    const auto* descriptor = query();
    if (!descriptor || descriptor->abi_version != project::game_module_abi_version ||
        descriptor->structure_size < sizeof(project::game_module_descriptor_v1) || !descriptor->engine_version ||
        engine_version != descriptor->engine_version || !descriptor->project_guid ||
        project_guid != descriptor->project_guid || !descriptor->module_id || module_id != descriptor->module_id ||
        descriptor->kind != project::game_module_kind_v1::editor || !descriptor->start || !descriptor->stop)
    {
        close_candidate();
        return {.classification = module_reload_classification::native_host_restart_required,
                .message = "Project editor module identity, role, or ABI is incompatible"};
    }
    std::string schema_error;
    auto next_components = copy_components(*descriptor, schema_error);
    if (!schema_error.empty())
    {
        close_candidate();
        return {.message = std::move(schema_error)};
    }
    auto next_registrations = copy_registrations(*descriptor, schema_error);
    if (!schema_error.empty())
    {
        close_candidate();
        return {.message = std::move(schema_error)};
    }
    const auto reload_classification = classify(components_, next_components);
    if (reload_classification == module_reload_classification::native_host_restart_required && loaded())
    {
        close_candidate();
        return {.classification = reload_classification, .generation = generation_,
                .message = classification_message(reload_classification)};
    }
    if (loaded() && prepare_reload_ && !prepare_reload_())
    {
        close_candidate();
        return {.generation = generation_, .message = "Project module could not quiesce its owned work"};
    }
    const project::game_module_host_v1 host{.log = module_log};
    const bool replacing = loaded();
    if (replacing && stop_) stop_();
    if (!descriptor->start(&host))
    {
        descriptor->stop();
        const bool restored = replacing && start_ && start_(&host);
        if (replacing && !restored)
        {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
            handle_ = nullptr;
            start_ = nullptr;
            prepare_reload_ = nullptr;
            stop_ = nullptr;
            components_.clear();
            registrations_.clear();
            std::error_code remove_error;
            std::filesystem::remove(loaded_path_, remove_error);
            loaded_path_.clear();
        }
        close_candidate();
        return {.classification = reload_classification, .generation = generation_,
                .message = restored
                               ? "Project module rejected startup; last-good generation restored"
                               : replacing ? "Project module rejected startup and the prior generation could not restart"
                                           : "Project module rejected startup"};
    }
    if (replacing)
    {
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
        std::error_code remove_error;
        std::filesystem::remove(loaded_path_, remove_error);
    }
    handle_ = candidate_handle;
    candidate_handle = nullptr;
    start_ = descriptor->start;
    prepare_reload_ = descriptor->prepare_reload;
    stop_ = descriptor->stop;
    generation_ = descriptor->generation ? descriptor->generation : next_generation;
    loaded_path_ = std::move(staged_path);
    components_ = std::move(next_components);
    registrations_ = std::move(next_registrations);
    return {.succeeded = true, .classification = reload_classification, .generation = generation_,
            .message = classification_message(reload_classification)};
}

void project_module_loader::unload() noexcept
{
    if (stop_) stop_();
    start_ = nullptr;
    stop_ = nullptr;
    prepare_reload_ = nullptr;
    if (handle_)
    {
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
    }
    handle_ = nullptr;
    components_.clear();
    registrations_.clear();
    if (!loaded_path_.empty())
    {
        std::error_code remove_error;
        std::filesystem::remove(loaded_path_, remove_error);
        loaded_path_.clear();
    }
}

} // namespace arc::editor
