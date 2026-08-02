#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace arc::editor
{

class project_module_loader
{
public:
    project_module_loader() = default;
    ~project_module_loader();
    project_module_loader(const project_module_loader&) = delete;
    project_module_loader& operator=(const project_module_loader&) = delete;

    [[nodiscard]] bool load(const std::filesystem::path& path, std::string_view engine_version,
                            std::string_view project_guid, std::string_view module_id, std::string& error);
    void unload() noexcept;
    [[nodiscard]] bool loaded() const noexcept { return handle_ != nullptr; }

private:
    void* handle_{};
    void (*stop_)(){};
};

} // namespace arc::editor
