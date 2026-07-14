#pragma once

#include <arc/editor/host_protocol.h>

#include <memory>

namespace arc::render
{
class renderer;
}

namespace arc::editor
{

struct editor_asset_state;
struct editor_scene_state;

class arc_host
{
public:
    explicit arc_host(std::unique_ptr<render::renderer> renderer);
    ~arc_host();

    arc_host(const arc_host&) = delete;
    arc_host& operator=(const arc_host&) = delete;

    host_response execute(const host_command_envelope& command);
    host_response execute(host_command_payload command);
    host_response open_project(
        const host_open_project_command& command,
        const editor_asset_state& assets,
        std::uint64_t request_id = 0);
    host_response query(const host_query_envelope& query) const;

    host_scene_snapshot scene_snapshot() const;
    host_selected_entity_snapshot selected_entity_snapshot() const;
    host_project_assets_snapshot project_assets_snapshot() const;
    std::optional<host_world_environment_snapshot> world_environment_snapshot(host_entity_id entity) const;
    std::vector<host_event> poll_events();

    host_viewport_frame request_viewport(const host_viewport_request& request);

    // Transitional escape hatches for panels that have not moved to protocol snapshots yet.
    render::renderer& renderer_for_legacy_clients() noexcept;
    const render::renderer& renderer_for_legacy_clients() const noexcept;
    editor_scene_state& scene_for_legacy_panels() noexcept;
    const editor_scene_state& scene_for_legacy_panels() const noexcept;

private:
    struct state;
    std::unique_ptr<state> state_;
};

class host_session
{
public:
    virtual ~host_session() = default;

    virtual host_response execute(const host_command_envelope& command) = 0;
    virtual host_response query(const host_query_envelope& query) = 0;
    virtual std::vector<host_event> poll_events() = 0;
    virtual host_viewport_frame request_viewport(const host_viewport_request& request) = 0;
};

class in_process_host_session final : public host_session
{
public:
    explicit in_process_host_session(std::shared_ptr<arc_host> host);

    host_response execute(const host_command_envelope& command) override;
    host_response query(const host_query_envelope& query) override;
    std::vector<host_event> poll_events() override;
    host_viewport_frame request_viewport(const host_viewport_request& request) override;

private:
    std::shared_ptr<arc_host> host_;
};

class stdio_host_session final
{
public:
    static std::string command_line(const host_command_envelope& command);
    static std::string query_line(const host_query_envelope& query);
};

class arc_host_manager
{
public:
    std::shared_ptr<arc_host> acquire(std::unique_ptr<render::renderer> renderer);

private:
    std::weak_ptr<arc_host> host_;
};

} // namespace arc::editor
