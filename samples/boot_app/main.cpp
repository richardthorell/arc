#include <arc/framework/framework.h>

#include <memory>

namespace
{

class boot_test_application final : public arc::application
{
public:
    arc::application_config configure() const override
    {
        arc::application_config config{};
        config.title = "ARC Boot Test";
        config.initial_width = 960;
        config.initial_height = 540;
        config.resizable = true;
        config.visible = true;
        return config;
    }
};

} // namespace

arc::application_ptr arc::create_application()
{
    return std::make_unique<boot_test_application>();
}
