#include <arc/framework/framework.h>

namespace
{

class smoke_application final : public arc::framework::application
{
public:
    arc::framework::application_config configure() const override
    {
        arc::framework::application_config config{};
        config.title = "ARC Windows Smoke";
        config.initial_width = 640;
        config.initial_height = 360;
        config.visible = false;
        return config;
    }

    void on_start() override {}
};

} // namespace

arc::framework::application_ptr arc::framework::create_application()
{
    return std::make_unique<smoke_application>();
}
