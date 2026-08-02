#include <arc/framework/framework.h>
#include <iostream>
#include <memory>
namespace { class server_application final : public arc::framework::application {}; }
int main()
{
    server_application application;
    arc::framework::headless_runtime_options options{};
    const auto result = arc::framework::run_headless(application, options);
    if (!result.succeeded) { std::cerr << result.error << '\n'; return 1; }
    return 0;
}
