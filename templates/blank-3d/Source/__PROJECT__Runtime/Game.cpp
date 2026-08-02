#include <arc/framework/framework.h>

#include <memory>

namespace
{
class game_application final : public arc::framework::application
{
public:
    arc::framework::application_config configure() const override
    {
        arc::framework::application_config config{};
        config.title = "{{PROJECT_NAME}}";
        return config;
    }
};
}

arc::framework::application_ptr arc::framework::create_application()
{
    return std::make_unique<game_application>();
}
