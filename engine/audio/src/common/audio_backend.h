#pragma once

#include <arc/audio/audio.h>

#include <memory>

namespace arc::audio::detail
{

class audio_backend
{
public:
    virtual ~audio_backend() = default;

    virtual runtime_error initialize(const runtime_config& config, device_id& playback_device) noexcept = 0;
    virtual void shutdown() noexcept = 0;
    [[nodiscard]] virtual bool initialized() const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<audio_backend> create_audio_backend(backend_type backend);

} // namespace arc::audio::detail
