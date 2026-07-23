#include <arc/framework/framework.h>

#include <charconv>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace
{

class headless_application final : public arc::application
{
public:
    arc::application_config configure() const override
    {
        arc::application_config config{};
        config.title = "ARC Headless Runtime";
        return config;
    }
};

bool parse_u64(std::string_view text, std::uint64_t& value)
{
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto [position, error] = std::from_chars(begin, end, value);
    return error == std::errc{} && position == end;
}

} // namespace

int main(int argc, char** argv)
{
    arc::headless_runtime_options options{};
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        if (argument == "--no-sleep")
            options.sleep_to_clock = false;
        else if (argument == "--client")
            options.default_world_role = arc::runtime_world_role::client;
        else if (argument == "--debug-time-controls")
            options.enable_debug_time_controls = true;
        else if (argument == "--ticks" && index + 1 < argc)
        {
            if (!parse_u64(argv[++index], options.maximum_ticks))
                return 2;
        }
        else if (argument == "--seed" && index + 1 < argc)
        {
            if (!parse_u64(argv[++index], options.seed))
                return 2;
        }
        else
        {
            std::cerr << "Unknown ARC headless option: " << argument << '\n';
            return 2;
        }
    }

    headless_application app;
    const arc::headless_runtime_result result = arc::run_headless(app, options);
    if (!result.succeeded)
    {
        std::cerr << result.error << '\n';
        return 1;
    }
    return 0;
}
