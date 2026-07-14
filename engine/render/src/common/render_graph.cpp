#include <arc/render/render_graph.h>

#include <algorithm>
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
    case render_format::rgba16_float: return 8;
    case render_format::rgba8_unorm:
    case render_format::rgba8_srgb:
    case render_format::rg16_float:
    case render_format::r32_uint:
    case render_format::d24_unorm_s8_uint:
    case render_format::d32_float: return 4;
    case render_format::r8_unorm: return 1;
    default: return 0;
    }
}

bool resources_compatible(const render_graph_resource& lhs, const render_graph_resource& rhs) noexcept
{
    return lhs.kind == rhs.kind &&
        lhs.format == rhs.format &&
        lhs.extent.width == rhs.extent.width &&
        lhs.extent.height == rhs.extent.height &&
        lhs.extent.depth == rhs.extent.depth &&
        lhs.extent_mode == rhs.extent_mode &&
        lhs.width_scale == rhs.width_scale &&
        lhs.height_scale == rhs.height_scale &&
        lhs.mip_levels == rhs.mip_levels &&
        lhs.array_layers == rhs.array_layers &&
        lhs.sample_count == rhs.sample_count;
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
    if (resource.name.empty())
        throw std::invalid_argument("render graph resource names must not be empty");
    if (find_resource(resource.name) != nullptr)
        throw std::invalid_argument("render graph resource names must be unique");

    const auto index = static_cast<std::uint32_t>(resources_.size());
    resources_.push_back(std::move(resource));
    return { index };
}

const render_graph_resource* render_graph::find_resource(std::string_view name) const noexcept
{
    for (const auto& resource : resources_)
    {
        if (resource.name == name)
            return &resource;
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

compiled_render_graph render_graph::compile() const
{
    struct resource_state
    {
        std::optional<std::uint32_t> last_writer;
        std::vector<std::uint32_t> readers;
        render_resource_usage last_usage{ render_resource_usage::unknown };
        std::uint32_t last_pass{};
        render_queue_type last_queue{ render_queue_type::graphics };
        bool used{};
    };

    std::vector<resource_state> resource_states(resources_.size());
    std::vector<std::vector<std::uint32_t>> edges(passes_.size());
    std::vector<render_resource_transition> transitions;

    const auto resolve_access = [&](const render_resource_access& access) {
        render_graph_resource_handle handle = access.handle;
        if (!handle.valid() && !access.resource.empty())
        {
            for (std::uint32_t index = 0; index < resources_.size(); ++index)
            {
                if (resources_[index].name == access.resource)
                {
                    handle = { index };
                    break;
                }
            }
        }
        if (!handle.valid() || handle.index >= resources_.size())
            throw std::invalid_argument("render graph access references an undeclared resource");
        const auto& resource = resources_[handle.index];
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

    const auto add_edge = [&](std::uint32_t before, std::uint32_t after) {
        if (before == after)
            return;
        auto& outgoing = edges[before];
        if (std::find(outgoing.begin(), outgoing.end(), after) == outgoing.end())
            outgoing.push_back(after);
    };

    const auto record_transition = [&](render_graph_resource_handle handle, const render_resource_access& access,
                                       std::uint32_t pass_index, render_queue_type queue) {
        auto& state = resource_states[handle.index];
        if (state.used && (state.last_usage != access.usage || state.last_queue != queue))
        {
            transitions.push_back({
                .handle = handle,
                .resource = resources_[handle.index].name,
                .before = state.last_usage,
                .after = access.usage,
                .before_pass = state.last_pass,
                .after_pass = pass_index,
                .before_queue = state.last_queue,
                .after_queue = queue
            });
        }
        state.last_usage = access.usage;
        state.last_pass = pass_index;
        state.last_queue = queue;
        state.used = true;
    };

    for (std::uint32_t index = 0; index < passes_.size(); ++index)
    {
        const auto& pass = passes_[index];
        if (pass.name.empty())
            throw std::invalid_argument("render graph pass names must not be empty");

        std::optional<std::uint32_t> attachment_samples;
        const auto validate_attachment = [&](const render_resource_access& access) {
            if (access.usage != render_resource_usage::color_attachment &&
                access.usage != render_resource_usage::depth_attachment)
                return;
            const auto handle = resolve_access(access);
            const auto samples = resources_[handle.index].sample_count;
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
            if (read.write)
                throw std::invalid_argument("render graph reads must not be marked writable");
            const auto handle = resolve_access(read);
            auto& state = resource_states[handle.index];
            if (!state.last_writer && !resources_[handle.index].imported)
                throw std::invalid_argument("internal render graph resource is read before its first write");
            if (state.last_writer)
                add_edge(*state.last_writer, index); // RAW
            if (std::find(state.readers.begin(), state.readers.end(), index) == state.readers.end())
                state.readers.push_back(index);
            record_transition(handle, read, index, pass.queue);
        }

        for (const auto& write : pass.writes)
        {
            if (!write.write)
                throw std::invalid_argument("render graph writes must be marked writable");
            const auto handle = resolve_access(write);
            auto& state = resource_states[handle.index];
            if (state.last_writer)
                add_edge(*state.last_writer, index); // WAW
            for (const auto reader : state.readers)
                add_edge(reader, index); // WAR
            state.readers.clear();
            state.last_writer = index;
            record_transition(handle, write, index, pass.queue);
        }
    }

    std::vector<std::uint32_t> indegree(passes_.size());
    for (const auto& outgoing : edges)
    {
        for (const auto after : outgoing)
            ++indegree[after];
    }

    std::deque<std::uint32_t> ready;
    for (std::uint32_t index = 0; index < indegree.size(); ++index)
    {
        if (indegree[index] == 0)
            ready.push_back(index);
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
            if (--indegree[after] == 0)
                ready.push_back(after);
        }
    }
    if (order.size() != passes_.size())
        throw std::invalid_argument("render graph dependency cycle detected");

    compiled_render_graph result;
    result.passes.reserve(order.size());
    result.resources = resources_;
    result.transitions = std::move(transitions);

    std::vector<std::uint32_t> compiled_index(passes_.size());
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
        result.passes.push_back({
            .source_index = index,
            .name = pass.name,
            .queue = pass.queue,
            .kind = pass.kind,
            .reads = std::move(reads),
            .writes = std::move(writes),
            .record = pass.record,
            .user_data = pass.user_data
        });
    }

    for (auto& transition : result.transitions)
    {
        transition.before_pass = compiled_index[transition.before_pass];
        transition.after_pass = compiled_index[transition.after_pass];
    }

    result.lifetimes.resize(resources_.size());
    std::vector<std::uint32_t> physical_last_pass;
    std::vector<std::uint32_t> physical_resource_owner;
    for (std::uint32_t index = 0; index < resources_.size(); ++index)
    {
        auto& lifetime = result.lifetimes[index];
        lifetime.handle = { index };
        const auto& resource = resources_[index];
        const auto bytes_per_pixel = format_bytes_per_pixel(resource.format);
        lifetime.estimated_bytes = bytes_per_pixel * resource.extent.width * resource.extent.height * resource.extent.depth *
            resource.array_layers * resource.sample_count;
    }

    for (std::uint32_t pass_index = 0; pass_index < result.passes.size(); ++pass_index)
    {
        const auto update_lifetime = [&](const render_resource_access& access) {
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
        const auto& resource = resources_[index];
        bool assigned = false;
        if (!resource.imported && !resource.persistent && lifetime.first_pass != render_graph_resource_handle::invalid_index)
        {
            for (std::uint32_t physical = 0; physical < physical_last_pass.size(); ++physical)
            {
                if (physical_last_pass[physical] < lifetime.first_pass &&
                    resources_compatible(resource, resources_[physical_resource_owner[physical]]))
                {
                    lifetime.physical_resource = physical;
                    physical_last_pass[physical] = lifetime.last_pass;
                    physical_resource_owner[physical] = index;
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

    return result;
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
    if (target.empty())
        target = "viewport";

    render_graph graph;
    const auto target_resource = graph.add_resource({
        .name = target,
        .kind = render_resource_kind::color_texture,
        .format = render_format::rgba16_float,
        .persistent = true
    });
    graph.add_pass({
        .name = "clear " + target,
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::clear,
        .writes = { {
            .handle = target_resource,
            .kind = render_resource_kind::color_texture,
            .usage = render_resource_usage::color_attachment,
            .write = true,
            .load_op = render_load_op::clear,
            .store_op = render_store_op::store
        } }
    });
    graph.add_pass({
        .name = "present " + target,
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::present,
        .reads = { {
            .handle = target_resource,
            .kind = render_resource_kind::color_texture,
            .usage = render_resource_usage::sampled,
            .write = false
        } }
    });
    return graph;
}

std::string_view render_format_name(render_format format) noexcept
{
    switch (format)
    {
    case render_format::rgba8_unorm: return "rgba8";
    case render_format::rgba8_srgb: return "rgba8_srgb";
    case render_format::rgba16_float: return "rgba16f";
    case render_format::rg16_float: return "rg16f";
    case render_format::r8_unorm: return "r8";
    case render_format::r32_uint: return "r32ui";
    case render_format::d24_unorm_s8_uint: return "d24s8";
    case render_format::d32_float: return "d32f";
    default: return "unknown";
    }
}

} // namespace arc::render
