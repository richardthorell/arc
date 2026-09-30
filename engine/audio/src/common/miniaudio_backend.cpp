#include "audio_backend.h"

#include <miniaudio.h>

#include <memory>

namespace arc::audio::detail
{
namespace
{

constexpr device_id primary_playback_device_id = 1;

ma_format to_miniaudio_format(const sample_format format) noexcept
{
    switch (format)
    {
    case sample_format::signed_16:
        return ma_format_s16;
    case sample_format::signed_24:
        return ma_format_s24;
    case sample_format::float_32:
        return ma_format_f32;
    }

    return ma_format_unknown;
}

void silence_callback(ma_device* device, void* output, const void*, const ma_uint32 frame_count) noexcept
{
    if (device == nullptr || output == nullptr)
    {
        return;
    }

    ma_silence_pcm_frames(output, frame_count, device->playback.format, device->playback.channels);
}

class miniaudio_backend final : public audio_backend
{
public:
    ~miniaudio_backend() override
    {
        shutdown();
    }

    runtime_error initialize(const runtime_config& config, device_id& playback_device) noexcept override
    {
        shutdown();
        playback_device = invalid_device_id;

        ma_context_config context_config = ma_context_config_init();
        ma_result result = MA_ERROR;
        if (config.device == device_mode::null_output)
        {
            const ma_backend backends[] = {ma_backend_null};
            result = ma_context_init(backends, 1, &context_config, &context_);
        }
        else
        {
            result = ma_context_init(nullptr, 0, &context_config, &context_);
        }

        if (result != MA_SUCCESS)
        {
            return runtime_error::backend_initialization_failed;
        }
        context_initialized_ = true;

        ma_device_config device_config = ma_device_config_init(ma_device_type_playback);
        device_config.playback.format = to_miniaudio_format(config.output.format);
        device_config.playback.channels = config.output.channel_count;
        device_config.sampleRate = config.output.sample_rate;
        device_config.dataCallback = silence_callback;

        result = ma_device_init(&context_, &device_config, &device_);
        if (result != MA_SUCCESS)
        {
            shutdown();
            return runtime_error::device_initialization_failed;
        }
        device_initialized_ = true;

        result = ma_device_start(&device_);
        if (result != MA_SUCCESS)
        {
            shutdown();
            return runtime_error::device_start_failed;
        }
        device_started_ = true;

        playback_device = primary_playback_device_id;
        return runtime_error::none;
    }

    void shutdown() noexcept override
    {
        if (device_started_)
        {
            static_cast<void>(ma_device_stop(&device_));
            device_started_ = false;
        }

        if (device_initialized_)
        {
            ma_device_uninit(&device_);
            device_initialized_ = false;
        }

        if (context_initialized_)
        {
            ma_context_uninit(&context_);
            context_initialized_ = false;
        }
    }

    [[nodiscard]] bool initialized() const noexcept override
    {
        return device_started_;
    }

private:
    ma_context context_{};
    ma_device device_{};
    bool context_initialized_ = false;
    bool device_initialized_ = false;
    bool device_started_ = false;
};

} // namespace

std::unique_ptr<audio_backend> create_audio_backend(const backend_type backend)
{
    switch (backend)
    {
    case backend_type::miniaudio:
        return std::make_unique<miniaudio_backend>();
    }

    return {};
}

} // namespace arc::audio::detail
