#include "project_module_loader.h"

#include <arc/diagnostics/diagnostics.h>
#include <arc/project/project_module.h>

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
} // namespace

project_module_loader::~project_module_loader()
{
    unload();
}

bool project_module_loader::load(const std::filesystem::path& path, std::string_view engine_version,
                                 std::string_view project_guid, std::string_view module_id, std::string& error)
{
    unload();
#if defined(_WIN32)
    HMODULE library = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                                           LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!library)
    {
        error = "project editor module could not be loaded";
        return false;
    }
    auto query = reinterpret_cast<project::query_editor_module_v1>(GetProcAddress(library, "arc_query_editor_module_v1"));
    handle_ = library;
#else
    void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library)
    {
        const char* diagnostic = dlerror();
        error = diagnostic ? diagnostic : "project editor module could not be loaded";
        return false;
    }
    auto query = reinterpret_cast<project::query_editor_module_v1>(dlsym(library, "arc_query_editor_module_v1"));
    handle_ = library;
#endif
    if (!query)
    {
        error = "project editor module does not export ABI version 1";
        unload();
        return false;
    }
    const auto* descriptor = query();
    if (!descriptor || descriptor->abi_version != project::editor_module_abi_version ||
        !descriptor->engine_version || engine_version != descriptor->engine_version ||
        !descriptor->project_guid || project_guid != descriptor->project_guid ||
        !descriptor->module_id || module_id != descriptor->module_id || !descriptor->start || !descriptor->stop)
    {
        error = "project editor module identity or ABI is incompatible";
        unload();
        return false;
    }
    const project::editor_module_host_v1 host{.log = module_log};
    if (!descriptor->start(&host))
    {
        error = "project editor module rejected startup";
        unload();
        return false;
    }
    stop_ = descriptor->stop;
    return true;
}

void project_module_loader::unload() noexcept
{
    if (stop_) stop_();
    stop_ = nullptr;
    if (!handle_) return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
    handle_ = nullptr;
}

} // namespace arc::editor
