#include "android_package_assets.h"

namespace arc::platform::android
{
bool valid_package_asset_path(std::string_view path) noexcept
{
    if (path.empty() || path.front() == '/' || path.back() == '/' ||
        path.find('\\') != path.npos || path.find(':') != path.npos) return false;
    std::size_t begin = 0;
    while (begin < path.size())
    {
        const auto end = path.find('/', begin);
        const auto component = path.substr(begin, end == path.npos ? path.size() - begin : end - begin);
        if (component.empty() || component == "." || component == ".." ||
            component.back() == '.' || component.back() == ' ') return false;
        for (const unsigned char c : component)
            if (c < 32 || c == 127) return false;
        if (end == path.npos) break;
        begin = end + 1;
    }
    return true;
}
} // namespace arc::platform::android
