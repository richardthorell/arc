#include "android_project_runtime_module.h"

#include <arc/diagnostics/log.h>

#include <dlfcn.h>

#include <exception>
#include <string>
#include <string_view>

#ifndef ARC_ANDROID_ENGINE_VERSION
#error "ARC_ANDROID_ENGINE_VERSION must be defined for native project module validation"
#endif

namespace arc::platform::android
{
namespace
{
constexpr auto library_name = "libarc_project_runtime.so";

void report_module_log(void*, const char* category, const char* message)
{
    arc::diagnostics::info(category ? category : "Android.Project",
                           message ? message : "Project module emitted an empty message");
}

[[nodiscard]] bool valid_descriptor(const arc::project::game_module_descriptor_v1* descriptor)
{
    using namespace arc::project;
    if (!descriptor || descriptor->abi_version != game_module_abi_version ||
        descriptor->structure_size != sizeof(game_module_descriptor_v1))
        return false;
    if (!descriptor->engine_version || std::string_view(descriptor->engine_version) != ARC_ANDROID_ENGINE_VERSION)
        return false;
    if (!descriptor->project_guid || !*descriptor->project_guid || !descriptor->module_id || !*descriptor->module_id)
        return false;
    if (descriptor->kind != game_module_kind_v1::runtime || !descriptor->start || !descriptor->stop) return false;
    if ((descriptor->component_count && !descriptor->components) ||
        (descriptor->registration_count && !descriptor->registrations))
        return false;
    return true;
}
} // namespace

android_project_runtime_module::~android_project_runtime_module() noexcept
{
    shutdown();
}

bool android_project_runtime_module::load(bool required)
{
    if (handle_) return descriptor_ != nullptr;

    // dlopen by SONAME stays within the application's Android linker namespace.
    handle_ = dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
    if (!handle_)
    {
        if (!required)
        {
            arc::diagnostics::info("Android.Project", "No project native module (engine host smoke mode)");
            return true;
        }
        const auto* failure = dlerror();
        std::string message = "Required project native module could not load: ";
        message += (failure ? failure : library_name);
        arc::diagnostics::error("Android.Project", message);
        return false;
    }

    dlerror();
    void* symbol = dlsym(handle_, "arc_query_game_module_v1");
    const auto query = reinterpret_cast<arc::project::query_game_module_v1>(symbol);
    const auto* symbol_error = dlerror();
    if (symbol_error || !query)
    {
        arc::diagnostics::error("Android.Project", "Project library lacks arc_query_game_module_v1");
        shutdown();
        return false;
    }
    try
    {
        descriptor_ = query();
    }
    catch (...)
    {
        arc::diagnostics::error("Android.Project", "Project module descriptor threw an exception");
        shutdown();
        return false;
    }
    if (!valid_descriptor(descriptor_))
    {
        arc::diagnostics::error("Android.Project", "Project module ABI, identity or lifecycle contract is invalid");
        descriptor_ = nullptr;
        shutdown();
        return false;
    }

    project_guid_ = descriptor_->project_guid;
    arc::diagnostics::info("Android.Project",
                           "Project native runtime validated: " + std::string(descriptor_->module_id));
    return true;
}

bool android_project_runtime_module::start()
{
    if (!descriptor_) return true;
    if (started_) return true;
    arc::project::game_module_host_v1 host;
    host.log = &report_module_log;
    try
    {
        if (!descriptor_->start(&host))
        {
            arc::diagnostics::error("Android.Project", "Project runtime module rejected startup");
            return false;
        }
    }
    catch (const std::exception& e)
    {
        arc::diagnostics::error("Android.Project", std::string("Project runtime startup failed: ") + e.what());
        return false;
    }
    catch (...)
    {
        arc::diagnostics::error("Android.Project", "Project runtime startup threw an unknown exception");
        return false;
    }
    started_ = true;
    return true;
}

void android_project_runtime_module::shutdown() noexcept
{
    if (started_ && descriptor_ && descriptor_->stop)
    {
        try
        {
            descriptor_->stop();
        }
        catch (...)
        {
            arc::diagnostics::error("Android.Project", "Project runtime stop threw an exception");
        }
    }
    started_ = false;
    descriptor_ = nullptr;
    project_guid_.clear();
    if (handle_)
    {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

} // namespace arc::platform::android
