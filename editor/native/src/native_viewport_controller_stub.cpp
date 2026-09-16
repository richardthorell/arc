#include "native_viewport_controller.h"

#include <iostream>
#include <utility>

namespace arc::editor
{
namespace
{

class unavailable_native_viewport_controller final : public native_viewport_controller
{
public:
    void attach(std::string, std::uint64_t, std::int32_t, std::int32_t, std::uint32_t, std::uint32_t) override
    {
        std::cerr << "arc_host_process native viewport rendering is not available in this build\n";
    }

    bool create_shared(std::string, std::uint64_t, std::uint32_t, std::uint32_t, std::string& error) override
    {
        error = "Shared viewport rendering is not available in this build";
        return false;
    }

    void release_frame(std::string, std::uint64_t, std::uint64_t, std::string) override {}

    void set_visible(std::string_view, bool) override {}

    void pointer(const host_viewport_pointer_command&) override {}

    void key(const host_viewport_key_command&) override {}

    void resize(std::string_view, std::int32_t, std::int32_t, std::uint32_t, std::uint32_t) override {}

    void detach(std::string_view) override {}

    void stop() override {}
};

} // namespace

std::unique_ptr<native_viewport_controller> make_native_viewport_controller(std::shared_ptr<arc_host>, std::mutex&,
                                                                            std::mutex&, jobs::job_system&)
{
    return std::make_unique<unavailable_native_viewport_controller>();
}

} // namespace arc::editor
