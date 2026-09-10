#include "project_module_loader.h"
#include "project_runtime_components.h"

#include <arc/diagnostics/diagnostics.h>
#include <arc/framework/runtime_world.h>

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <nlohmann/json.hpp>

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

project_runtime_component_value* runtime_project_component(void* user_data, project::game_entity_v1 entity,
                                                           const char* component_id) noexcept
{
    if (!user_data || !component_id || !*component_id || !entity.valid()) return nullptr;
    auto& context = *static_cast<ecs::system_context*>(user_data);
    const ecs::entity native_entity{entity.index, entity.generation};
    auto& world = context.owner();
    if (!world.alive(native_entity)) return nullptr;
    auto* components = world.try_get<project_runtime_component_set>(native_entity);
    return components ? find_project_runtime_component(*components, component_id) : nullptr;
}

bool has_runtime_project_component(void* user_data, project::game_entity_v1 entity, const char* component_id) noexcept
{
    return runtime_project_component(user_data, entity, component_id) != nullptr;
}

const char* read_runtime_project_component_json(void* user_data, project::game_entity_v1 entity,
                                                const char* component_id) noexcept
{
    const auto* component = runtime_project_component(user_data, entity, component_id);
    return component ? component->json.c_str() : nullptr;
}

bool patch_runtime_project_component_json(void* user_data, project::game_entity_v1 entity, const char* component_id,
                                          const char* patch_json) noexcept
{
    auto* component = runtime_project_component(user_data, entity, component_id);
    if (!component || !patch_json) return false;
    auto patch = nlohmann::json::parse(patch_json, nullptr, false);
    if (!patch.is_object()) return false;
    auto current = nlohmann::json::parse(component->json, nullptr, false);
    if (!current.is_object()) current = nlohmann::json::object();
    current.update(patch);
    current["typeId"] = component->stable_id;
    current["version"] = component->schema_version;
    component->json = current.dump();
    return true;
}

bool stable_component_id(std::string_view value)
{
    return value.size() == 32 && std::all_of(value.begin(), value.end(),
                                             [](unsigned char character) { return std::isxdigit(character) != 0; });
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
        if (std::any_of(result.begin(), result.end(),
                        [&](const auto& current) { return current.stable_id == source.stable_id; }))
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
                std::any_of(component.fields.begin(), component.fields.end(),
                            [&](const auto& current) { return current.stable_id == field.stable_id; }))
            {
                error = "project module contains an invalid or duplicate field stable ID";
                return {};
            }
            component.fields.push_back(
                {.stable_id = field.stable_id,
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
                 .asset_type_restriction = field.asset_type_restriction ? field.asset_type_restriction : "",
                 .entity_component_restriction =
                     field.entity_component_restriction ? field.entity_component_restriction : ""});
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
            std::any_of(result.begin(), result.end(),
                        [&](const auto& current) { return current.stable_id == source.stable_id; }))
        {
            error = "project module contains an invalid or duplicate registration ID";
            return {};
        }
        result.push_back({source.kind, source.stable_id, source.name});
    }
    return result;
}

bool valid_system_phase(project::game_system_phase_v1 phase) noexcept
{
    switch (phase)
    {
        case project::game_system_phase_v1::input:
        case project::game_system_phase_v1::network_receive:
        case project::game_system_phase_v1::gameplay_commands:
        case project::game_system_phase_v1::movement:
        case project::game_system_phase_v1::physics:
        case project::game_system_phase_v1::abilities:
        case project::game_system_phase_v1::ai:
        case project::game_system_phase_v1::replication:
        case project::game_system_phase_v1::presentation_extraction:
            return true;
    }
    return false;
}

bool valid_system_priority(project::game_system_priority_v1 priority) noexcept
{
    switch (priority)
    {
        case project::game_system_priority_v1::critical:
        case project::game_system_priority_v1::high:
        case project::game_system_priority_v1::normal:
        case project::game_system_priority_v1::low:
        case project::game_system_priority_v1::background:
            return true;
    }
    return false;
}

ecs::system_phase to_system_phase(project::game_system_phase_v1 phase) noexcept
{
    switch (phase)
    {
        case project::game_system_phase_v1::input:
            return ecs::system_phase::input;
        case project::game_system_phase_v1::network_receive:
            return ecs::system_phase::network_receive;
        case project::game_system_phase_v1::gameplay_commands:
            return ecs::system_phase::gameplay_commands;
        case project::game_system_phase_v1::movement:
            return ecs::system_phase::movement;
        case project::game_system_phase_v1::physics:
            return ecs::system_phase::physics;
        case project::game_system_phase_v1::abilities:
            return ecs::system_phase::abilities;
        case project::game_system_phase_v1::ai:
            return ecs::system_phase::ai;
        case project::game_system_phase_v1::replication:
            return ecs::system_phase::replication;
        case project::game_system_phase_v1::presentation_extraction:
            return ecs::system_phase::presentation_extraction;
    }
    return ecs::system_phase::gameplay_commands;
}

jobs::job_priority to_job_priority(project::game_system_priority_v1 priority) noexcept
{
    switch (priority)
    {
        case project::game_system_priority_v1::critical:
            return jobs::job_priority::critical;
        case project::game_system_priority_v1::high:
            return jobs::job_priority::high;
        case project::game_system_priority_v1::normal:
            return jobs::job_priority::normal;
        case project::game_system_priority_v1::low:
            return jobs::job_priority::low;
        case project::game_system_priority_v1::background:
            return jobs::job_priority::background;
    }
    return jobs::job_priority::normal;
}

std::vector<project_system_registration> copy_system_registrations(const project::game_module_descriptor_v1& descriptor,
                                                                   std::string& error)
{
    std::vector<project_system_registration> result;
    for (std::size_t index = 0; index < descriptor.registration_count; ++index)
    {
        const auto& registration = descriptor.registrations[index];
        if (registration.kind != project::game_registration_kind_v1::ecs_system) continue;
        if (!registration.descriptor)
        {
            error = "project ECS system registration is missing its executable descriptor";
            return {};
        }
        const auto& system = *static_cast<const project::game_system_descriptor_v1*>(registration.descriptor);
        if (system.structure_size < sizeof(project::game_system_descriptor_v1) || !system.execute ||
            !valid_system_phase(system.phase) || !valid_system_priority(system.priority) ||
            (system.before_count && !system.before) || (system.after_count && !system.after))
        {
            error = "project module contains an invalid ECS system descriptor";
            return {};
        }

        project_system_registration copied{.stable_id = registration.stable_id,
                                           .name = registration.name,
                                           .phase = system.phase,
                                           .priority = system.priority,
                                           .user_data = system.user_data,
                                           .execute = system.execute};
        copied.before.reserve(system.before_count);
        copied.after.reserve(system.after_count);
        for (std::size_t dependency = 0; dependency < system.before_count; ++dependency)
        {
            if (!system.before[dependency] || !*system.before[dependency])
            {
                error = "project ECS system contains an invalid before dependency";
                return {};
            }
            copied.before.emplace_back(system.before[dependency]);
        }
        for (std::size_t dependency = 0; dependency < system.after_count; ++dependency)
        {
            if (!system.after[dependency] || !*system.after[dependency])
            {
                error = "project ECS system contains an invalid after dependency";
                return {};
            }
            copied.after.emplace_back(system.after[dependency]);
        }
        result.push_back(std::move(copied));
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
            const auto field =
                std::find_if(component->fields.begin(), component->fields.end(),
                             [&](const auto& candidate) { return candidate.stable_id == old_field.stable_id; });
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
        case module_reload_classification::initial_load:
            return "Project module loaded";
        case module_reload_classification::safe_hot_reload:
            return "Project module hot reloaded";
        case module_reload_classification::play_session_restart_required:
            return "Module loaded; play session restart required";
        case module_reload_classification::native_host_restart_required:
            return "Module schema requires a native host restart";
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
                                                            std::string_view project_guid, std::string_view module_id,
                                                            bool /*is_reload*/)
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
    if (library)
        query = reinterpret_cast<project::query_game_module_v1>(GetProcAddress(library, "arc_query_game_module_v1"));
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
    auto next_systems = copy_system_registrations(*descriptor, schema_error);
    if (!schema_error.empty())
    {
        close_candidate();
        return {.message = std::move(schema_error)};
    }
    const auto reload_classification = classify(components_, next_components);
    if (reload_classification == module_reload_classification::native_host_restart_required && loaded())
    {
        close_candidate();
        return {.classification = reload_classification,
                .generation = generation_,
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
            systems_.clear();
            std::error_code remove_error;
            std::filesystem::remove(loaded_path_, remove_error);
            loaded_path_.clear();
        }
        close_candidate();
        return {.classification = reload_classification,
                .generation = generation_,
                .message = restored    ? "Project module rejected startup; last-good generation restored"
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
    systems_ = std::move(next_systems);
    return {.succeeded = true,
            .classification = reload_classification,
            .generation = generation_,
            .message = classification_message(reload_classification)};
}

project_system_install_result project_module_loader::install_systems(framework::runtime_world& world) const
{
    project_system_install_result result{.succeeded = true};
    for (const auto& source : systems_)
    {
        ecs::system_descriptor descriptor{.name = source.stable_id,
                                          .phase = to_system_phase(source.phase),
                                          .priority = to_job_priority(source.priority),
                                          // The v1 project-system ABI intentionally starts conservative. A native
                                          // module can access the complete ECS world through native_context, so
                                          // serialize it against other world-mutating systems until explicit
                                          // component access declarations are added to the module ABI.
                                          .exclusive_world_access = true,
                                          .before = source.before,
                                          .after = source.after,
                                          .execute = [execute = source.execute, user_data = source.user_data,
                                                      stable_id = source.stable_id](ecs::system_context& native_context)
                                          {
                                              project::game_system_context_v1 context{
                                                  .native_context = &native_context,
                                                  .tick_id = native_context.tick_id().value,
                                                  .world_id = native_context.world_id(),
                                                  .delta_seconds = native_context.delta_seconds(),
                                                  .fixed_delta_seconds = native_context.fixed_delta_seconds(),
                                                  .frame_delta_seconds = native_context.frame_delta_seconds(),
                                                  .interpolation_alpha = native_context.interpolation_alpha(),
                                                  .presentation = native_context.presentation(),
                                                  .project_component_user_data = &native_context,
                                                  .has_project_component = has_runtime_project_component,
                                                  .read_project_component_json = read_runtime_project_component_json,
                                                  .patch_project_component_json = patch_runtime_project_component_json,
                                              };
                                              if (!execute(user_data, &context))
                                                  throw std::runtime_error("project ECS system '" + stable_id +
                                                                           "' reported execution failure");
                                          }};
        if (!world.systems().add(std::move(descriptor)))
        {
            result.succeeded = false;
            result.error = "Could not install project ECS system '" + source.stable_id + "'";
            return result;
        }
        ++result.installed;
    }
    return result;
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
    systems_.clear();
    if (!loaded_path_.empty())
    {
        std::error_code remove_error;
        std::filesystem::remove(loaded_path_, remove_error);
        loaded_path_.clear();
    }
}

} // namespace arc::editor
