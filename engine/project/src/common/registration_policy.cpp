#include <arc/project/registration_policy.h>

namespace arc::project
{

bool valid_registration_kind(game_registration_kind_v1 kind) noexcept
{
    switch (kind)
    {
        case game_registration_kind_v1::ecs_system:
        case game_registration_kind_v1::service:
        case game_registration_kind_v1::asset_type:
        case game_registration_kind_v1::importer:
        case game_registration_kind_v1::cook_processor:
        case game_registration_kind_v1::console_command:
        case game_registration_kind_v1::editor_extension:
        case game_registration_kind_v1::play_lifecycle:
            return true;
    }
    return false;
}

bool registration_kind_allowed(game_module_kind_v1 module_kind, game_registration_kind_v1 registration_kind) noexcept
{
    if (!valid_registration_kind(registration_kind)) return false;

    switch (module_kind)
    {
        case game_module_kind_v1::editor:
            return true;
        case game_module_kind_v1::runtime:
        case game_module_kind_v1::server:
            switch (registration_kind)
            {
                case game_registration_kind_v1::ecs_system:
                case game_registration_kind_v1::service:
                case game_registration_kind_v1::asset_type:
                case game_registration_kind_v1::console_command:
                case game_registration_kind_v1::play_lifecycle:
                    return true;
                case game_registration_kind_v1::importer:
                case game_registration_kind_v1::cook_processor:
                case game_registration_kind_v1::editor_extension:
                    return false;
            }
            return false;
    }
    return false;
}

std::string_view registration_kind_name(game_registration_kind_v1 kind) noexcept
{
    switch (kind)
    {
        case game_registration_kind_v1::ecs_system:
            return "ecs_system";
        case game_registration_kind_v1::service:
            return "service";
        case game_registration_kind_v1::asset_type:
            return "asset_type";
        case game_registration_kind_v1::importer:
            return "importer";
        case game_registration_kind_v1::cook_processor:
            return "cook_processor";
        case game_registration_kind_v1::console_command:
            return "console_command";
        case game_registration_kind_v1::editor_extension:
            return "editor_extension";
        case game_registration_kind_v1::play_lifecycle:
            return "play_lifecycle";
    }
    return "unknown";
}

} // namespace arc::project
