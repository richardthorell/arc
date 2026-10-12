#include <arc/scene/irradiance_probe_asset.h>

namespace arc::scene
{
namespace
{

class irradiance_probe_volume_importer final : public assets::asset_importer
{
public:
    irradiance_probe_volume_importer()
    {
        descriptor_.id = assets::importer_ids::irradiance_probe_volume;
        descriptor_.name = "ARC Irradiance Probe Volume";
        descriptor_.version = 1u;
        descriptor_.settings_version = 1u;
        descriptor_.extensions = {".arcprobe"};
        descriptor_.output_types = {assets::asset_types::irradiance_probe_volume};
    }

    const assets::asset_importer_descriptor& descriptor() const noexcept override { return descriptor_; }

    assets::asset_import_result import(const assets::asset_import_context& context) override
    {
        if (context.cancellation.stop_requested())
            return {.error = {.code = assets::asset_error_code::cancelled,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = "Irradiance probe-volume import was cancelled"}};

        const auto decoded = render::decode_irradiance_probe_volume(context.source_bytes);
        if (!decoded)
            return {.error = {.code = assets::asset_error_code::import_failed,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = decoded.error}};

        auto volume = std::make_shared<render::irradiance_probe_volume_descriptor>(decoded.volume);
        const std::size_t resident_bytes =
            sizeof(render::irradiance_probe_volume_descriptor) +
            volume->probes.size() * sizeof(render::irradiance_probe_sample);

        assets::asset_import_result result;
        result.payload = assets::asset_payload::make<render::irradiance_probe_volume_descriptor>(
            assets::asset_types::irradiance_probe_volume, std::move(volume), resident_bytes);
        result.residency = assets::asset_residency::cpu;
        return result;
    }

private:
    assets::asset_importer_descriptor descriptor_;
};

} // namespace

std::unique_ptr<assets::asset_importer> make_irradiance_probe_volume_importer()
{
    return std::make_unique<irradiance_probe_volume_importer>();
}

bool register_irradiance_probe_volume_importer(assets::asset_manager& manager)
{
    return manager.register_importer(make_irradiance_probe_volume_importer());
}

irradiance_probe_volume_binding_result
refresh_irradiance_probe_volume_binding(irradiance_probe_volume_component& component,
                                        assets::asset_manager& manager)
{
    irradiance_probe_volume_binding_result result;
    if (!component.asset.guid.valid() && component.asset.path_hint.empty())
    {
        component.runtime_volume.reset();
        component.asset_generation = 0;
        result.succeeded = true;
        result.message = "No baked irradiance probe-volume asset assigned";
        return result;
    }

    auto reference = component.asset;
    reference.expected_type = assets::asset_types::irradiance_probe_volume;
    if (!reference.guid.valid() && !reference.path_hint.empty())
        reference = manager.resolve(reference.path_hint, assets::asset_types::irradiance_probe_volume);
    if (!reference.guid.valid())
    {
        component.runtime_volume.reset();
        result.message = "Irradiance probe-volume asset reference could not be resolved";
        return result;
    }

    auto pending = manager.load<render::irradiance_probe_volume_descriptor>(
        {.reference = reference,
         .priority = assets::asset_streaming_priority::high,
         .residency = assets::asset_residency::cpu,
         .allow_fallback = false});
    auto loaded = pending.get();
    if (!loaded || !loaded.asset.get())
    {
        component.runtime_volume.reset();
        result.message = loaded ? "Irradiance probe-volume payload has the wrong runtime type"
                                : (loaded.error.message.empty() ? "Irradiance probe-volume asset could not be loaded"
                                                               : loaded.error.message);
        return result;
    }

    component.asset.guid = loaded.asset.requested_guid();
    component.asset.expected_type = assets::asset_types::irradiance_probe_volume;
    if (const auto snapshot = manager.find(loaded.asset.resolved_guid()))
        component.asset.path_hint = assets::normalize_asset_path(snapshot->source_path);
    component.asset_generation = loaded.asset.generation();
    component.runtime_volume = std::make_shared<render::irradiance_probe_volume_descriptor>(*loaded.asset.get());

    result.succeeded = true;
    result.bound = true;
    result.generation = component.asset_generation;
    result.message = "Baked irradiance probe-volume asset loaded";
    return result;
}

} // namespace arc::scene
