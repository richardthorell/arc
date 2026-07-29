#include <arc/framework/framework.h>

#include <windows.h>

#include <memory>
#include <string_view>

namespace
{

class sample_game_application final : public arc::framework::application
{
public:
    arc::framework::application_config configure() const override
    {
        arc::framework::application_config config{};
        config.title = "ARC Sample Game";
        config.initial_width = 960;
        config.initial_height = 540;
        config.resizable = true;
        config.visible = !ci_smoke_;
        return config;
    }

    void on_update(const arc::framework::frame_time&) override
    {
        if (ci_smoke_ && ++frame_count_ >= 3) PostQuitMessage(0);
    }

private:
    const bool ci_smoke_{std::wstring_view(GetCommandLineW()).find(L"--ci-smoke") != std::wstring_view::npos};
    std::uint32_t frame_count_{};
};

} // namespace

arc::framework::application_ptr arc::framework::create_application()
{
    return std::make_unique<sample_game_application>();
}
