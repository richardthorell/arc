#pragma once

#include <arc/project/project_module.h>

#include <string>

namespace arc::platform::android
{

// Process-owned native project module. The APK library and its callbacks must
// outlive the runtime/session and must not be unloaded on surface recreation.
class android_project_runtime_module final
{
public:
    android_project_runtime_module() = default;
    ~android_project_runtime_module() noexcept;

    android_project_runtime_module(const android_project_runtime_module&) = delete;
    android_project_runtime_module& operator=(const android_project_runtime_module&) = delete;

    // An engine-only smoke fixture may omit the project module. A staged
    // distribution must always carry the expected project runtime library.
    [[nodiscard]] bool load(bool required);
    [[nodiscard]] bool start();
    void shutdown() noexcept;

    [[nodiscard]] bool loaded() const noexcept
    {
        return descriptor_ != nullptr;
    }
    [[nodiscard]] const std::string& project_guid() const noexcept
    {
        return project_guid_;
    }

private:
    void* handle_{};
    const arc::project::game_module_descriptor_v1* descriptor_{};
    std::string project_guid_;
    bool started_{};
};

} // namespace arc::platform::android
