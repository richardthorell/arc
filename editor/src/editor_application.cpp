#include <arc/framework/framework.h>
#include <arc/diagnostics/log.h>
#include <memory>

namespace
{

class editor_application final : public arc::application
{
public:
    arc::application_config configure() const override
    {
        arc::application_config config{};
        config.title = "ARC Editor";
        config.initial_width = 1440;
        config.initial_height = 900;
        config.resizable = true;
        config.visible = true;
        config.start_focused = true;
        config.maximized = true;
        return config;
    }

    void on_start() override
    {
        arc::info("editor", "Editor runtime started");
    }

    void register_modules(arc::module_registry&) override
    {
        arc::info("editor", "Editor render host owns the active renderer backend");
    }

    void on_shutdown() override
    {
        arc::info("editor", "Editor runtime shutdown");
    }
};

} // namespace

namespace arc
{

application_ptr create_application()
{
    return std::make_unique<editor_application>();
}

} // namespace arc
