#pragma once

#include <arc/assets/assets.h>
#include <arc/render/lighting.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace arc::scene
{

struct [[nodiscard]] photometric_profile_artifact_result
{
    render::photometric_profile profile;
    std::string error;

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return error.empty();
    }
};

[[nodiscard]] std::vector<std::byte> encode_photometric_profile_artifact(const render::photometric_profile& profile);

[[nodiscard]] photometric_profile_artifact_result decode_photometric_profile_artifact(std::span<const std::byte> bytes);

[[nodiscard]] std::unique_ptr<assets::asset_importer> make_photometric_profile_importer();
bool register_photometric_profile_importer(assets::asset_manager& manager);

} // namespace arc::scene
