#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace arc::jobs
{
class job_system;
}

namespace arc::editor
{
class arc_host;
struct host_viewport_key_command;
struct host_viewport_pointer_command;

class native_viewport_controller
{
public:
    virtual ~native_viewport_controller() = default;

    virtual void attach(std::string viewport_id, std::uint64_t native_handle, std::int32_t x, std::int32_t y,
                        std::uint32_t width, std::uint32_t height) = 0;
    virtual bool create_shared(std::string viewport_id, std::uint64_t consumer_process_id, std::uint32_t width,
                               std::uint32_t height, std::string& error) = 0;
    virtual void release_frame(std::string viewport_id, std::uint64_t generation, std::uint64_t frame_id,
                               std::string consumer_handle) = 0;
    virtual void set_visible(std::string_view viewport_id, bool visible) = 0;
    virtual void pointer(const host_viewport_pointer_command& pointer) = 0;
    virtual void key(const host_viewport_key_command& key) = 0;
    virtual void resize(std::string_view viewport_id, std::int32_t x, std::int32_t y, std::uint32_t width,
                        std::uint32_t height) = 0;
    virtual void detach(std::string_view viewport_id) = 0;
    virtual void stop() = 0;
};

std::unique_ptr<native_viewport_controller> make_native_viewport_controller(std::shared_ptr<arc_host> host,
                                                                            std::mutex& host_mutex,
                                                                            std::mutex& output_mutex,
                                                                            jobs::job_system& jobs);

} // namespace arc::editor
