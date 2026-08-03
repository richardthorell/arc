#pragma once

#include <arc/project/project_module.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace arc::editor
{

struct project_field_schema
{
    std::uint64_t stable_id{};
    std::string name;
    std::string display_name;
    std::string category;
    std::string tooltip;
    project::game_field_kind_v1 kind{};
    project::game_field_flags_v1 flags{};
    std::string default_json;
    double minimum{};
    double maximum{};
    bool has_minimum{};
    bool has_maximum{};
    std::string asset_type_restriction;
    std::string entity_component_restriction;
};

struct project_component_schema
{
    std::string stable_id;
    std::string canonical_name;
    std::string display_name;
    std::string category;
    std::string tooltip;
    std::uint32_t schema_version{1};
    std::vector<project_field_schema> fields;
};

struct project_registration_schema
{
    project::game_registration_kind_v1 kind{};
    std::string stable_id;
    std::string name;
};

enum class module_reload_classification : std::uint8_t
{
    initial_load,
    safe_hot_reload,
    play_session_restart_required,
    native_host_restart_required
};

struct module_reload_result
{
    bool succeeded{};
    module_reload_classification classification{module_reload_classification::initial_load};
    std::uint64_t generation{};
    std::string message;
};

class project_module_loader
{
public:
    project_module_loader() = default;
    ~project_module_loader();
    project_module_loader(const project_module_loader&) = delete;
    project_module_loader& operator=(const project_module_loader&) = delete;

    [[nodiscard]] module_reload_result load(const std::filesystem::path& path, std::string_view engine_version,
                                            std::string_view project_guid, std::string_view module_id);
    [[nodiscard]] module_reload_result reload(const std::filesystem::path& path, std::string_view engine_version,
                                              std::string_view project_guid, std::string_view module_id);
    void unload() noexcept;
    [[nodiscard]] bool loaded() const noexcept { return handle_ != nullptr; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] const std::vector<project_component_schema>& component_schemas() const noexcept
    {
        return components_;
    }
    [[nodiscard]] const std::vector<project_registration_schema>& registrations() const noexcept
    {
        return registrations_;
    }

private:
    [[nodiscard]] module_reload_result load_generation(const std::filesystem::path& path,
                                                       std::string_view engine_version,
                                                       std::string_view project_guid,
                                                       std::string_view module_id,
                                                       bool is_reload);

    void* handle_{};
    bool (*start_)(const project::game_module_host_v1*){};
    bool (*prepare_reload_)(){};
    void (*stop_)(){};
    std::uint64_t generation_{};
    std::filesystem::path loaded_path_;
    std::vector<project_component_schema> components_;
    std::vector<project_registration_schema> registrations_;
};

} // namespace arc::editor
