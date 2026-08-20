#include <arc/render/render_graph.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace arc::render
{

namespace
{

std::uint64_t format_bytes_per_pixel(render_format format) noexcept
{
    switch (format)
    {
        case render_format::rgba16_float:
            return 8;
        case render_format::rgba8_unorm:
        case render_format::rgba8_srgb:
        case render_format::rg16_float:
        case render_format::r32_uint:
        case render_format::r32_float:
        case render_format::d24_unorm_s8_uint:
        case render_format::d32_float:
            return 4;
        case render_format::r8_unorm:
            return 1;
        default:
            return 0;
    }
}

bool resources_compatible(const render_graph_resource& lhs, const render_graph_resource& rhs) noexcept
{
    if (lhs.kind == render_resource_kind::buffer || rhs.kind == render_resource_kind::buffer)
        return lhs.kind == rhs.kind && lhs.memory == rhs.memory && lhs.byte_size == rhs.byte_size &&
               lhs.element_stride == rhs.element_stride;
    return lhs.kind == rhs.kind && lhs.dimension == rhs.dimension && lhs.format == rhs.format &&
           lhs.memory == rhs.memory && lhs.extent.width == rhs.extent.width &&
           lhs.extent.height == rhs.extent.height && lhs.extent.depth == rhs.extent.depth &&
           lhs.extent_mode == rhs.extent_mode && lhs.width_scale == rhs.width_scale &&
           lhs.height_scale == rhs.height_scale && lhs.mip_levels == rhs.mip_levels &&
           lhs.array_layers == rhs.array_layers && lhs.sample_count == rhs.sample_count;
}

bool usage_matches_resource(render_resource_kind kind, render_resource_usage usage) noexcept
{
    switch (usage)
    {
        case render_resource_usage::color_attachment:
            return kind == render_resource_kind::color_texture || kind == render_resource_kind::swapchain_image;
        case render_resource_usage::depth_attachment:
            return kind == render_resource_kind::depth_texture;
        case render_resource_usage::present:
            return kind == render_resource_kind::swapchain_image || kind == render_resource_kind::color_texture;
        case render_resource_usage::uniform_buffer:
        case render_resource_usage::storage_buffer:
        case render_resource_usage::indirect_buffer:
        case render_resource_usage::vertex_buffer:
        case render_resource_usage::index_buffer:
            return kind == render_resource_kind::buffer;
        case render_resource_usage::sampled:
        case render_resource_usage::storage:
        case render_resource_usage::transfer_src:
        case render_resource_usage::transfer_dst:
            return kind != render_resource_kind::unknown;
        default:
            return false;
    }
}

} // namespace

render_graph_resource_handle render_graph::add_resource(render_graph_resource resource)
{
    if (resource.name.empty()) throw std::invalid_argument("render graph resource names must not be empty");
    if (find_resource(resource.name) != nullptr)
        throw std::invalid_argument("render graph resource names must be unique");
    if (resource.history_length == 0) throw std::invalid_argument("render graph history length must be positive");
    if (resource.history_length > 1) resource.persistent = true;
    if (resource.persistent && resource.persistent_key.empty()) resource.persistent_key = resource.name;
    if (resource.imported)
        resource.lifetime = render_resource_lifetime_class::external;
    else if (resource.persistent && resource.lifetime == render_resource_lifetime_class::transient)
        resource.lifetime = render_resource_lifetime_class::per_view;
    if (resource.lifetime != render_resource_lifetime_class::transient ||
        resource.memory != render_memory_class::device_local)
        resource.allow_aliasing = false;

    const auto index = static_cast<std::uint32_t>(resources_.size());
    resources_.push_back(std::move(resource));
    return {index};
}

const render_graph_resource* render_graph::find_resource(std::string_view name) const noexcept
{
    for (const auto& resource : resources_)
    {
        if (resource.name == name) return &resource;
    }
    return nullptr;
}

const render_graph_resource* render_graph::find_resource(render_graph_resource_handle handle) const noexcept
{
    return handle.valid() && handle.index < resources_.size() ? &resources_[handle.index] : nullptr;
}

std::uint32_t render_graph::add_pass(render_graph_pass pass)
{
    const auto index = static_cast<std::uint32_t>(passes_.size());
    passes_.push_back(std::move(pass));
    return index;
}

render_graph_compile_result render_graph::compile(const render_graph_compile_options& options) const
{
    std::string active_pass;
    std::string active_resource;
    try
    {
    struct resource_state
    {
        std::optional<std::uint32_t> last_writer;
        std::vector<std::uint32_t> readers;
        render_resource_usage last_usage{render_resource_usage::unknown};
        std::uint32_t last_pass{};
        render_queue_type last_queue{render_queue_type::graphics};
        render_pipeline_stage last_stages{render_pipeline_stage::none};
        render_subresource_range last_subresources{};
        bool last_write{};
        bool used{};
    };

    std::vector<render_graph_resource> resolved_resources = resources_;
    for (auto& resource : resolved_resources)
    {
        if (resource.extent_mode != render_extent_mode::relative_to_view) continue;
        const auto base = options.render_extent.width != 0 && options.render_extent.height != 0
                              ? options.render_extent
                              : options.output_extent;
        if (base.width == 0 || base.height == 0) continue;
        resource.extent.width = std::max(
            1u, static_cast<std::uint32_t>(std::ceil(static_cast<float>(base.width) * resource.width_scale)));
        resource.extent.height = std::max(
            1u, static_cast<std::uint32_t>(std::ceil(static_cast<float>(base.height) * resource.height_scale)));
        resource.extent.depth = std::max(resource.extent.depth, 1u);
    }

    const auto resolve_queue = [&](render_queue_type queue)
    {
        if (queue == render_queue_type::compute && !options.compute_queue_available)
            return render_queue_type::graphics;
        if (queue == render_queue_type::transfer && !options.transfer_queue_available)
            return render_queue_type::graphics;
        return queue;
    };

    std::vector<resource_state> resource_states(resources_.size());
    std::vector<resource_state> previous_resource_states(resources_.size());
    std::vector<std::vector<std::uint32_t>> edges(passes_.size());
    std::vector<render_resource_transition> transitions;

    const auto resolve_access = [&](const render_resource_access& access)
    {
        render_graph_resource_handle handle = access.handle;
        if (!handle.valid() && !access.resource.empty())
        {
            for (std::uint32_t index = 0; index < resources_.size(); ++index)
            {
                if (resources_[index].name == access.resource)
                {
                    handle = {index};
                    break;
                }
            }
        }
        if (!handle.valid() || handle.index >= resources_.size())
            throw std::invalid_argument("render graph access references an undeclared resource");
        const auto& resource = resolved_resources[handle.index];
        active_resource = resource.name;
        if (!access.resource.empty() && access.resource != resource.name)
            throw std::invalid_argument("render graph resource handle and name disagree");
        if (access.kind != render_resource_kind::unknown && access.kind != resource.kind)
            throw std::invalid_argument("render graph access kind does not match its resource");
        if (access.usage == render_resource_usage::unknown)
            throw std::invalid_argument("render graph accesses must declare a usage");
        if (!usage_matches_resource(resource.kind, access.usage))
            throw std::invalid_argument("render graph access usage is incompatible with its resource");
        return handle;
    };

    const auto add_edge = [&](std::uint32_t before, std::uint32_t after)
    {
        if (before == after) return;
        auto& outgoing = edges[before];
        if (std::find(outgoing.begin(), outgoing.end(), after) == outgoing.end()) outgoing.push_back(after);
    };

    const auto record_transition = [&](render_graph_resource_handle handle, const render_resource_access& access,
                                       std::uint32_t pass_index, render_queue_type queue)
    {
        auto& state = access.history == render_history_access::previous ? previous_resource_states[handle.index]
                                                                        : resource_states[handle.index];
        const auto stages = access.stages != render_pipeline_stage::none
                                ? access.stages
                                : (queue == render_queue_type::compute
                                       ? render_pipeline_stage::compute_shader
                                       : queue == render_queue_type::transfer ? render_pipeline_stage::transfer
                                                                             : render_pipeline_stage::all_graphics);
        if (state.used && (state.last_usage != access.usage || state.last_queue != queue || state.last_write ||
                           access.write || state.last_stages != stages))
        {
            transitions.push_back({.handle = handle,
                                   .resource = resources_[handle.index].name,
                                   .before = state.last_usage,
                                   .after = access.usage,
                                   .before_history = access.history,
                                   .after_history = access.history,
                                   .before_stages = state.last_stages,
                                   .after_stages = stages,
                                   .subresources = access.subresources,
                                   .release = state.last_queue != queue,
                                   .acquire = state.last_queue != queue,
                                   .before_pass = state.last_pass,
                                   .after_pass = pass_index,
                                   .before_queue = state.last_queue,
                                   .after_queue = queue});
        }
        state.last_usage = access.usage;
        state.last_pass = pass_index;
        state.last_queue = queue;
        state.last_stages = stages;
        state.last_subresources = access.subresources;
        state.last_write = access.write;
        state.used = true;
    };

    for (std::uint32_t index = 0; index < passes_.size(); ++index)
    {
        const auto& pass = passes_[index];
        const auto pass_queue = resolve_queue(pass.queue);
        active_pass = pass.name;
        if (pass.name.empty()) throw std::invalid_argument("render graph pass names must not be empty");

        std::optional<std::uint32_t> attachment_samples;
        const auto validate_attachment = [&](const render_resource_access& access)
        {
            if (access.usage != render_resource_usage::color_attachment &&
                access.usage != render_resource_usage::depth_attachment)
                return;
            const auto handle = resolve_access(access);
            const auto samples = resolved_resources[handle.index].sample_count;
            if (attachment_samples && *attachment_samples != samples)
                throw std::invalid_argument("render graph pass attachments must use the same sample count");
            attachment_samples = samples;
        };
        for (const auto& read : pass.reads)
            validate_attachment(read);
        for (const auto& write : pass.writes)
            validate_attachment(write);

        for (const auto& read : pass.reads)
        {
            if (read.write) throw std::invalid_argument("render graph reads must not be marked writable");
            const auto handle = resolve_access(read);
            if (read.history == render_history_access::previous &&
                (!resources_[handle.index].persistent || resources_[handle.index].history_length < 2))
                throw std::invalid_argument("previous history reads require a persistent resource with history");
            auto& state = resource_states[handle.index];
            if (read.history == render_history_access::current && !state.last_writer &&
                !resources_[handle.index].imported)
                throw std::invalid_argument("internal render graph resource is read before its first write");
            if (read.history == render_history_access::current)
            {
                if (state.last_writer) add_edge(*state.last_writer, index); // RAW
                if (std::find(state.readers.begin(), state.readers.end(), index) == state.readers.end())
                    state.readers.push_back(index);
            }
            record_transition(handle, read, index, pass_queue);
        }

        for (const auto& write : pass.writes)
        {
            if (!write.write) throw std::invalid_argument("render graph writes must be marked writable");
            if (write.history != render_history_access::current)
                throw std::invalid_argument("render graph passes may only write the current history generation");
            const auto handle = resolve_access(write);
            auto& state = resource_states[handle.index];
            if (state.last_writer) add_edge(*state.last_writer, index); // WAW
            for (const auto reader : state.readers)
                add_edge(reader, index); // WAR
            state.readers.clear();
            state.last_writer = index;
            record_transition(handle, write, index, pass_queue);
        }
    }

    std::vector<std::vector<std::uint32_t>> predecessors(passes_.size());
    for (std::uint32_t before = 0; before < edges.size(); ++before)
        for (const auto after : edges[before])
            predecessors[after].push_back(before);

    std::vector<bool> live(passes_.size());
    std::vector<std::uint32_t> live_stack;
    for (std::uint32_t index = 0; index < passes_.size(); ++index)
    {
        const auto& pass = passes_[index];
        bool root = pass.side_effect || pass.kind == render_pass_kind::present;
        for (const auto& write : pass.writes)
        {
            const auto handle = resolve_access(write);
            const auto& resource = resolved_resources[handle.index];
            root = root || resource.exported || resource.imported || resource.persistent ||
                   resource.lifetime != render_resource_lifetime_class::transient;
        }
        if (root)
        {
            live[index] = true;
            live_stack.push_back(index);
        }
    }
    while (!live_stack.empty())
    {
        const auto index = live_stack.back();
        live_stack.pop_back();
        for (const auto predecessor : predecessors[index])
            if (!live[predecessor])
            {
                live[predecessor] = true;
                live_stack.push_back(predecessor);
            }
    }

    const auto live_count = static_cast<std::size_t>(std::count(live.begin(), live.end(), true));
    std::vector<std::uint32_t> indegree(passes_.size());
    for (std::uint32_t before = 0; before < edges.size(); ++before)
    {
        if (!live[before]) continue;
        for (const auto after : edges[before])
            if (live[after]) ++indegree[after];
    }

    std::deque<std::uint32_t> ready;
    for (std::uint32_t index = 0; index < indegree.size(); ++index)
    {
        if (live[index] && indegree[index] == 0) ready.push_back(index);
    }

    std::vector<std::uint32_t> order;
    order.reserve(passes_.size());
    while (!ready.empty())
    {
        const auto index = ready.front();
        ready.pop_front();
        order.push_back(index);
        for (const auto after : edges[index])
        {
            if (live[after] && --indegree[after] == 0) ready.push_back(after);
        }
    }
    if (order.size() != live_count) throw std::invalid_argument("render graph dependency cycle detected");

    compiled_render_graph result;
    result.view = options;
    result.passes.reserve(order.size());
    result.resources = resolved_resources;
    for (std::uint32_t index = 0; index < passes_.size(); ++index)
        if (!live[index]) result.culled_passes.push_back({.source_index = index, .name = passes_[index].name});

    std::vector<std::uint32_t> compiled_index(passes_.size(), render_graph_resource_handle::invalid_index);
    for (std::uint32_t output_index = 0; output_index < order.size(); ++output_index)
    {
        const auto index = order[output_index];
        compiled_index[index] = output_index;
        const auto& pass = passes_[index];
        auto reads = pass.reads;
        auto writes = pass.writes;
        for (auto& read : reads)
        {
            read.handle = resolve_access(read);
            read.resource = resources_[read.handle.index].name;
        }
        for (auto& write : writes)
        {
            write.handle = resolve_access(write);
            write.resource = resources_[write.handle.index].name;
        }
        result.passes.push_back({.source_index = index,
                                 .name = pass.name,
                                 .queue = resolve_queue(pass.queue),
                                 .kind = pass.kind,
                                 .builtin = pass.builtin,
                                 .reads = std::move(reads),
                                 .writes = std::move(writes),
                                 .record = pass.record,
                                 .payload = pass.payload});
    }

    for (auto& transition : transitions)
    {
        if (!live[transition.before_pass] || !live[transition.after_pass]) continue;
        transition.before_pass = compiled_index[transition.before_pass];
        transition.after_pass = compiled_index[transition.after_pass];
        result.transitions.push_back(std::move(transition));
    }

    result.lifetimes.resize(resources_.size());
    std::vector<std::uint32_t> physical_last_pass;
    std::vector<std::uint32_t> physical_resource_owner;
    for (std::uint32_t index = 0; index < resources_.size(); ++index)
    {
        auto& lifetime = result.lifetimes[index];
        lifetime.handle = {index};
        const auto& resource = resolved_resources[index];
        if (resource.kind == render_resource_kind::buffer)
            lifetime.estimated_bytes = resource.byte_size != 0
                                           ? resource.byte_size
                                           : static_cast<std::uint64_t>(resource.extent.width) *
                                                 std::max<std::uint32_t>(resource.element_stride, 1u);
        else
        {
            const auto bytes_per_pixel = format_bytes_per_pixel(resource.format);
            const auto layers = static_cast<std::uint64_t>(resource.array_layers) *
                                (resource.dimension == render_texture_dimension::texture_cube ? 6u : 1u);
            std::uint64_t mip_bytes{};
            auto width = std::max(resource.extent.width, 1u);
            auto height = std::max(resource.extent.height, 1u);
            auto depth = std::max(resource.extent.depth, 1u);
            for (std::uint32_t mip = 0; mip < std::max(resource.mip_levels, 1u); ++mip)
            {
                mip_bytes += bytes_per_pixel * width * height * depth;
                width = std::max(width / 2u, 1u);
                height = std::max(height / 2u, 1u);
                if (resource.dimension == render_texture_dimension::texture_3d) depth = std::max(depth / 2u, 1u);
            }
            lifetime.estimated_bytes = mip_bytes * layers * resource.sample_count;
        }
    }

    for (std::uint32_t pass_index = 0; pass_index < result.passes.size(); ++pass_index)
    {
        const auto update_lifetime = [&](const render_resource_access& access)
        {
            auto& lifetime = result.lifetimes[access.handle.index];
            lifetime.first_pass = std::min(lifetime.first_pass, pass_index);
            lifetime.last_pass = std::max(lifetime.last_pass, pass_index);
        };
        for (const auto& read : result.passes[pass_index].reads)
            update_lifetime(read);
        for (const auto& write : result.passes[pass_index].writes)
            update_lifetime(write);
    }

    for (std::uint32_t index = 0; index < resources_.size(); ++index)
    {
        auto& lifetime = result.lifetimes[index];
        const auto& resource = resolved_resources[index];
        if (lifetime.first_pass == render_graph_resource_handle::invalid_index)
        {
            lifetime.physical_resource = render_graph_resource_handle::invalid_index;
            continue;
        }
        bool assigned = false;
        if (!resource.imported && !resource.persistent && resource.allow_aliasing &&
            resource.memory == render_memory_class::device_local &&
            lifetime.first_pass != render_graph_resource_handle::invalid_index)
        {
            for (std::uint32_t physical = 0; physical < physical_last_pass.size(); ++physical)
            {
                if (physical_last_pass[physical] < lifetime.first_pass &&
                    resources_compatible(resource, resolved_resources[physical_resource_owner[physical]]))
                {
                    lifetime.physical_resource = physical;
                    physical_last_pass[physical] = lifetime.last_pass;
                    physical_resource_owner[physical] = index;
                    lifetime.aliased = true;
                    assigned = true;
                    break;
                }
            }
        }
        if (!assigned)
        {
            lifetime.physical_resource = static_cast<std::uint32_t>(physical_last_pass.size());
            physical_last_pass.push_back(lifetime.last_pass);
            physical_resource_owner.push_back(index);
        }
    }

    std::vector<std::uint32_t> pass_submission(result.passes.size());
    std::uint64_t queue_signals[3]{};
    const auto queue_slot = [](render_queue_type queue) {
        return queue == render_queue_type::graphics ? 0u : queue == render_queue_type::compute ? 1u : 2u;
    };
    for (std::uint32_t pass_index = 0; pass_index < result.passes.size(); ++pass_index)
    {
        const auto queue = result.passes[pass_index].queue;
        if (result.submissions.empty() || result.submissions.back().queue != queue)
        {
            const auto signal = ++queue_signals[queue_slot(queue)];
            result.submissions.push_back({.queue = queue, .signal_value = signal});
        }
        const auto submission_index = static_cast<std::uint32_t>(result.submissions.size() - 1);
        result.submissions.back().passes.push_back(pass_index);
        pass_submission[pass_index] = submission_index;
    }

    for (const auto& transition : result.transitions)
    {
        if (transition.before_queue == transition.after_queue) continue;
        const auto producer = pass_submission[transition.before_pass];
        const auto consumer = pass_submission[transition.after_pass];
        if (producer == consumer) continue;
        const render_queue_wait wait{.queue = transition.before_queue,
                                     .value = result.submissions[producer].signal_value};
        auto& waits = result.submissions[consumer].waits;
        const bool duplicate = std::any_of(waits.begin(), waits.end(), [&](const render_queue_wait& existing)
                                           { return existing.queue == wait.queue && existing.value == wait.value; });
        if (!duplicate) waits.push_back(wait);
    }

    for (std::uint32_t index = 0; index < resources_.size(); ++index)
    {
        const auto& resource = resolved_resources[index];
        if (!resource.persistent) continue;
        result.history_rotations.push_back({.handle = {index},
                                            .persistent_key = resource.persistent_key,
                                            .history_length = resource.history_length,
                                            .reset = resource.history_reset,
                                            .invalidated =
                                                (resource.history_reset & options.temporal_reset) !=
                                                render_history_reset::none});
    }

    result.physical_resources.resize(physical_last_pass.size());
    for (std::uint32_t physical = 0; physical < result.physical_resources.size(); ++physical)
    {
        auto& allocation = result.physical_resources[physical];
        allocation.index = physical;
        allocation.kind = resolved_resources[physical_resource_owner[physical]].kind;
    }
    for (const auto& lifetime : result.lifetimes)
    {
        if (lifetime.physical_resource >= result.physical_resources.size()) continue;
        auto& allocation = result.physical_resources[lifetime.physical_resource];
        allocation.estimated_bytes = std::max(allocation.estimated_bytes, lifetime.estimated_bytes);
        allocation.logical_resources.push_back(lifetime.handle);
    }

    return render_graph_compile_result::success(std::move(result));
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        auto code = render_graph_compile_error_code::invalid_access;
        if (message.find("cycle") != std::string::npos)
            code = render_graph_compile_error_code::dependency_cycle;
        else if (message.find("read before") != std::string::npos)
            code = render_graph_compile_error_code::read_before_write;
        else if (message.find("history") != std::string::npos)
            code = render_graph_compile_error_code::invalid_history;
        else if (message.find("attachment") != std::string::npos)
            code = render_graph_compile_error_code::invalid_attachment;
        else if (message.find("resource") != std::string::npos)
            code = render_graph_compile_error_code::invalid_resource;
        return render_graph_compile_result::failure(
            {.code = code, .message = message, .pass = std::move(active_pass), .resource = std::move(active_resource)});
    }
}

void render_graph::clear()
{
    resources_.clear();
    passes_.clear();
}

const std::vector<render_graph_pass>& render_graph::passes() const noexcept
{
    return passes_;
}

const std::vector<render_graph_resource>& render_graph::resources() const noexcept
{
    return resources_;
}

render_graph make_clear_present_graph(std::string_view target_name)
{
    std::string target(target_name);
    if (target.empty()) target = "viewport";

    render_graph graph;
    const auto target_resource = graph.add_resource({.name = target,
                                                     .kind = render_resource_kind::color_texture,
                                                     .format = render_format::rgba16_float,
                                                     .persistent = true});
    graph.add_pass({.name = "clear " + target,
                    .queue = render_queue_type::graphics,
                    .kind = render_pass_kind::clear,
                    .writes = {{.handle = target_resource,
                                .kind = render_resource_kind::color_texture,
                                .usage = render_resource_usage::color_attachment,
                                .write = true,
                                .load_op = render_load_op::clear,
                                .store_op = render_store_op::store}}});
    graph.add_pass({.name = "present " + target,
                    .queue = render_queue_type::graphics,
                    .kind = render_pass_kind::present,
                    .reads = {{.handle = target_resource,
                               .kind = render_resource_kind::color_texture,
                               .usage = render_resource_usage::sampled,
                               .write = false}}});
    return graph;
}

std::string_view render_format_name(render_format format) noexcept
{
    switch (format)
    {
        case render_format::rgba8_unorm:
            return "rgba8";
        case render_format::rgba8_srgb:
            return "rgba8_srgb";
        case render_format::rgba16_float:
            return "rgba16f";
        case render_format::rg16_float:
            return "rg16f";
        case render_format::r8_unorm:
            return "r8";
        case render_format::r32_uint:
            return "r32ui";
        case render_format::r32_float:
            return "r32f";
        case render_format::d24_unorm_s8_uint:
            return "d24s8";
        case render_format::d32_float:
            return "d32f";
        default:
            return "unknown";
    }
}

} // namespace arc::render
