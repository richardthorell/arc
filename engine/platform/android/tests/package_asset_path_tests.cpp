#include "../app/src/main/cpp/android_package_assets.h"
#include <cassert>

int main()
{
    using arc::platform::android::valid_package_asset_path;
    assert(valid_package_asset_path("arc-distribution.json"));
    assert(valid_package_asset_path("content/boot.arccookmanifest"));
    assert(valid_package_asset_path("content/Boot/0.arcpak"));
    for (const auto* path : {"", "../escape", "content/../escape", "content/./scene",
                             "/root", "C:/root", "content\\scene", "content//scene",
                             "content/", "content/\nscene"})
        assert(!valid_package_asset_path(path));
    return 0;
}
