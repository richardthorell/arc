#include <arc/render/render_graph.h>

#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace arc::render
{

std::uint32_t render_graph::add_resource(render_graph_resource resource)
{
    if (resource.name.empty())
        throw std::invalid_argument("render graph resource names must not be empty");
    if (find_resource(resource.name) != nullptr)
        throw std::invalid_argument("render graph resource names must be unique");

    const auto index = static_cast<std::uint32_t>(resources_.size());
    resources_.push_back(std::move(resource));
    return index;
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

std::uint32_t render_graph::add_pass(render_graph_pass pass)
{
    const auto index = static_cast<std::uint32_t>(passes_.size());
    passes_.push_back(std::move(pass));
    return index;
}

compiled_render_graph render_graph::compile() const
{
    std::unordered_map<std::string, std::uint32_t> last_writer;
    std::unordered_map<std::string, std::pair<render_resource_usage, std::uint32_t>> last_usage;
    std::vector<std::vector<std::uint32_t>> edges(passes_.size());
    std::vector<render_resource_transition> transitions;

    for (std::uint32_t index = 0; index < passes_.size(); ++index)
    {
        const auto& pass = passes_[index];
        if (pass.name.empty())
            throw std::invalid_argument("render graph pass names must not be empty");

        for (const auto& read : pass.reads)
        {
            if (read.resource.empty())
                throw std::invalid_argument("render graph read resources must not be empty");
            const auto writer = last_writer.find(read.resource);
            if (writer != last_writer.end())
                edges[writer->second].push_back(index);

            const auto last = last_usage.find(read.resource);
            if (last != last_usage.end() && last->second.first != read.usage)
            {
                transitions.push_back({
                    .resource = read.resource,
                    .before = last->second.first,
                    .after = read.usage,
                    .before_pass = last->second.second,
                    .after_pass = index
                });
            }
            last_usage[read.resource] = { read.usage, index };
        }

        for (const auto& write : pass.writes)
        {
            if (write.resource.empty())
                throw std::invalid_argument("render graph write resources must not be empty");
            const auto last = last_usage.find(write.resource);
            if (last != last_usage.end() && last->second.first != write.usage)
            {
                transitions.push_back({
                    .resource = write.resource,
                    .before = last->second.first,
                    .after = write.usage,
                    .before_pass = last->second.second,
                    .after_pass = index
                });
            }
            last_writer[write.resource] = index;
            last_usage[write.resource] = { write.usage, index };
        }
    }

    std::vector<int> state(passes_.size(), 0);
    std::vector<std::uint32_t> order;
    order.reserve(passes_.size());

    auto visit = [&](auto& self, std::uint32_t index) -> void {
        if (state[index] == 2)
            return;
        if (state[index] == 1)
            throw std::invalid_argument("render graph dependency cycle detected");

        state[index] = 1;
        for (const auto dependency : edges[index])
            self(self, dependency);
        state[index] = 2;
        order.push_back(index);
    };

    for (std::uint32_t index = 0; index < passes_.size(); ++index)
        visit(visit, index);

    compiled_render_graph result;
    result.passes.reserve(order.size());
    result.resources = resources_;
    result.transitions = std::move(transitions);

    for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator)
    {
        const auto index = *iterator;
        const auto& pass = passes_[index];
        result.passes.push_back({
            .source_index = index,
            .name = pass.name,
            .queue = pass.queue,
            .kind = pass.kind,
            .reads = pass.reads,
            .writes = pass.writes
        });
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
    graph.add_resource({
        .name = target,
        .kind = render_resource_kind::color_texture,
        .persistent = true
    });
    graph.add_pass({
        .name = "clear " + target,
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::clear,
        .writes = { {
            .resource = target,
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
            .resource = target,
            .kind = render_resource_kind::color_texture,
            .usage = render_resource_usage::sampled,
            .write = false
        } }
    });
    return graph;
}

} // namespace arc::render
