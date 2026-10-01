#include "audio_backend.h"

#include <utility>

namespace arc::audio
{

struct audio_runtime::implementation
{
    explicit implementation(const backend_type backend_type_value)
        : selected_backend(backend_type_value), backend(detail::create_audio_backend(backend_type_value))
    {
    }

    backend_type selected_backend = backend_type::miniaudio;
    std::unique_ptr<detail::audio_backend> backend;
    device_id playback_device = invalid_device_id;
};

audio_runtime::audio_runtime(const backend_type backend) : implementation_(std::make_unique<implementation>(backend)) {}

audio_runtime::~audio_runtime() = default;

audio_runtime::audio_runtime(audio_runtime&&) noexcept = default;

audio_runtime& audio_runtime::operator=(audio_runtime&&) noexcept = default;

runtime_error audio_runtime::initialize(const runtime_config& config) noexcept
{
    if (!implementation_ || !implementation_->backend)
    {
        return runtime_error::backend_unavailable;
    }

    shutdown();
    if (validate(config.output) != validation_error::none)
    {
        return runtime_error::invalid_configuration;
    }

    device_id playback_device = invalid_device_id;
    const runtime_error error = implementation_->backend->initialize(config, playback_device);
    if (error != runtime_error::none)
    {
        return error;
    }

    implementation_->playback_device = playback_device;
    return runtime_error::none;
}

void audio_runtime::shutdown() noexcept
{
    if (!implementation_ || !implementation_->backend)
    {
        return;
    }

    implementation_->backend->shutdown();
    implementation_->playback_device = invalid_device_id;
}

bool audio_runtime::initialized() const noexcept
{
    return implementation_ && implementation_->backend && implementation_->backend->initialized();
}

backend_type audio_runtime::backend() const noexcept
{
    return implementation_ ? implementation_->selected_backend : backend_type::miniaudio;
}

device_id audio_runtime::playback_device() const noexcept
{
    return implementation_ ? implementation_->playback_device : invalid_device_id;
}

} // namespace arc::audio
