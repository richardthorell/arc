from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    content = target.read_text()
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
    target.write_text(content.replace(old, new, 1))


project_module = "engine/project/inc/arc/project/project_module.h"
replace_once(
    project_module,
    "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n",
    "#pragma once\n\n#include <arc/project/runtime_world_api.h>\n\n#include <cstddef>\n#include <cstdint>\n",
)
replace_once(
    project_module,
    """/** @brief ABI-safe transient entity handle used by project runtime callbacks. */
struct game_entity_v1
{
    std::uint32_t index{0xffffffffu};
    std::uint32_t generation{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return index != 0xffffffffu;
    }
};

""",
    "",
)
replace_once(
    project_module,
    """    game_patch_project_component_json_v1 patch_project_component_json{};
    game_for_each_project_component_v1 for_each_project_component{};
};
""",
    """    game_patch_project_component_json_v1 patch_project_component_json{};
    game_for_each_project_component_v1 for_each_project_component{};
    const game_world_api_v1* world{}; ///< Stable runtime-world bridge valid only for this invocation.
};
""",
)
replace_once(
    project_module,
    """    void* user_data{};                ///< Module-owned state valid for the loaded generation.
    game_system_execute_v1 execute{}; ///< Called by ARC's ECS scheduler.
};
""",
    """    void* user_data{};                ///< Module-owned state valid for the loaded generation.
    game_system_execute_v1 execute{}; ///< Called by ARC's ECS scheduler.
    const game_core_component_access_v1* core_component_accesses{}; ///< Stable engine-component access declarations.
    std::size_t core_component_access_count{};                      ///< Number of entries in @ref core_component_accesses.
};
""",
)
replace_once(
    project_module,
    """struct game_play_context_v1
{
    std::size_t structure_size{sizeof(game_play_context_v1)};
    std::uint64_t world_id{}; ///< Runtime world that owns this Play session.
};
""",
    """struct game_play_context_v1
{
    std::size_t structure_size{sizeof(game_play_context_v1)};
    std::uint64_t world_id{};         ///< Runtime world that owns this Play session.
    const game_world_api_v1* world{}; ///< Stable runtime-world bridge for this lifecycle callback.
};
""",
)

loader_header = "editor/native/src/project_module_loader.h"
replace_once(
    loader_header,
    """    std::vector<project_system_component_access> component_accesses;
    bool unrestricted_native_world_access{true};
""",
    """    std::vector<project_system_component_access> component_accesses;
    std::vector<project::game_core_component_access_v1> core_component_accesses;
    bool unrestricted_native_world_access{true};
""",
)

loader = "editor/native/src/project_module_loader.cpp"
replace_once(
    loader,
    '#include "project_runtime_components.h"\n',
    '#include "project_runtime_components.h"\n#include "project_runtime_world_bridge.h"\n',
)
replace_once(
    loader,
    """struct project_play_session_guard
{
    project_play_lifecycle_registration lifecycle;
    project::game_play_context_v1 context;
    bool active{};
""",
    """struct project_play_session_guard
{
    project_play_lifecycle_registration lifecycle;
    runtime_world_bridge_context world_bridge;
    project::game_world_api_v1 world_api;
    project::game_play_context_v1 context;
    bool active{};
""",
)
replace_once(
    loader,
    """        const auto& system = *static_cast<const project::game_system_descriptor_v1*>(registration.descriptor);
        if (system.structure_size < sizeof(project::game_system_descriptor_v1) || !system.execute ||
            !valid_system_phase(system.phase) || !valid_system_priority(system.priority) ||
            (system.component_access_count && !system.component_accesses) || (system.before_count && !system.before) ||
            (system.after_count && !system.after))
""",
    """        const auto& system = *static_cast<const project::game_system_descriptor_v1*>(registration.descriptor);
        constexpr std::size_t base_descriptor_size = offsetof(project::game_system_descriptor_v1, core_component_accesses);
        const bool has_core_accesses = system.structure_size >= sizeof(project::game_system_descriptor_v1);
        if (system.structure_size < base_descriptor_size || !system.execute || !valid_system_phase(system.phase) ||
            !valid_system_priority(system.priority) || (system.component_access_count && !system.component_accesses) ||
            (has_core_accesses && system.core_component_access_count && !system.core_component_accesses) ||
            (system.before_count && !system.before) || (system.after_count && !system.after))
""",
)
replace_once(
    loader,
    """            copied.component_accesses.push_back({access.component_id, access.mode});
        }
        copied.before.reserve(system.before_count);
""",
    """            copied.component_accesses.push_back({access.component_id, access.mode});
        }
        if (has_core_accesses)
        {
            copied.core_component_accesses.reserve(system.core_component_access_count);
            for (std::size_t access_index = 0; access_index < system.core_component_access_count; ++access_index)
            {
                const auto access = system.core_component_accesses[access_index];
                if (!valid_core_component(access.component) || !valid_core_component_access_mode(access.mode))
                {
                    error = "project ECS system contains an invalid core component access declaration";
                    return {};
                }
                if (std::any_of(copied.core_component_accesses.begin(), copied.core_component_accesses.end(),
                                [&](const project::game_core_component_access_v1& existing)
                                { return existing.component == access.component; }))
                {
                    error = "project ECS system contains duplicate core component access declarations";
                    return {};
                }
                copied.core_component_accesses.push_back(access);
            }
        }
        copied.before.reserve(system.before_count);
""",
)
replace_once(
    loader,
    """        std::vector<ecs::component_access> scheduler_accesses;
        scheduler_accesses.reserve(source.component_accesses.size());
""",
    """        std::vector<ecs::component_access> scheduler_accesses;
        scheduler_accesses.reserve(source.component_accesses.size() + source.core_component_accesses.size());
""",
)
replace_once(
    loader,
    """            scheduler_accesses.push_back(
                {*component, access.mode == project::game_system_component_access_mode_v1::write
                                 ? ecs::component_access_mode::write
                                 : ecs::component_access_mode::read});
        }

        ecs::system_descriptor descriptor{
""",
    """            scheduler_accesses.push_back(
                {*component, access.mode == project::game_system_component_access_mode_v1::write
                                 ? ecs::component_access_mode::write
                                 : ecs::component_access_mode::read});
        }
        for (const auto& access : source.core_component_accesses)
            scheduler_accesses.push_back(scheduler_core_access(access));

        ecs::system_descriptor descriptor{
""",
)
replace_once(
    loader,
    """            .execute = [execute = source.execute, user_data = source.user_data, stable_id = source.stable_id,
                        declared_accesses = source.component_accesses,
                        unrestricted_native_world_access =
                            source.unrestricted_native_world_access](ecs::system_context& native_context)
            {
                project_component_bridge_context bridge{.native_context = &native_context,
                                                        .declared_accesses = &declared_accesses};
""",
    """            .execute = [execute = source.execute, user_data = source.user_data, stable_id = source.stable_id,
                        declared_accesses = source.component_accesses, core_accesses = source.core_component_accesses,
                        unrestricted_native_world_access =
                            source.unrestricted_native_world_access](ecs::system_context& native_context)
            {
                project_component_bridge_context bridge{.native_context = &native_context,
                                                        .declared_accesses = &declared_accesses};
                runtime_world_bridge_context world_bridge{.world = &native_context.owner(),
                                                          .commands = &native_context.commands(),
                                                          .declared_accesses = &core_accesses,
                                                          .unrestricted_access = unrestricted_native_world_access};
                auto world_api = make_runtime_world_api(world_bridge);
""",
)
replace_once(
    loader,
    """                    .patch_project_component_json = patch_runtime_project_component_json,
                    .for_each_project_component = for_each_runtime_project_component,
                };
                const bool succeeded = execute(user_data, &context);
                if (bridge.violation)
                    throw std::runtime_error("project ECS system '" + stable_id + "' violated declared " +
                                             bridge.violation_message);
                if (!succeeded)
""",
    """                    .patch_project_component_json = patch_runtime_project_component_json,
                    .for_each_project_component = for_each_runtime_project_component,
                    .world = &world_api,
                };
                const bool succeeded = execute(user_data, &context);
                if (bridge.violation)
                    throw std::runtime_error("project ECS system '" + stable_id + "' violated declared " +
                                             bridge.violation_message);
                if (world_bridge.violation)
                    throw std::runtime_error("project ECS system '" + stable_id + "' violated declared " +
                                             world_bridge.violation_message);
                if (!succeeded)
""",
)
replace_once(
    loader,
    """        auto guard = std::make_shared<project_play_session_guard>();
        guard->lifecycle = *play_lifecycle_;
        guard->context.world_id = world.id().value;
""",
    """        auto guard = std::make_shared<project_play_session_guard>();
        guard->lifecycle = *play_lifecycle_;
        guard->world_bridge.world = &world.entities();
        guard->world_bridge.unrestricted_access = true;
        guard->world_api = make_runtime_world_api(guard->world_bridge);
        guard->context.world_id = world.id().value;
        guard->context.world = &guard->world_api;
""",
)
