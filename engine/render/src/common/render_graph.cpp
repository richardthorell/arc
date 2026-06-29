#include <arc/render/render_graph.h>

#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace arc::render
{

std::uint32_t render_graph::add_pass(render_graph_pass pass)
{
    const auto index = static_cast<std::uint32_t>(passes_.size());
    passes_.push_back(std::move(pass));
    return index;
}

compiled_render_graph render_graph::compile() const
{
    std::unordered_map<std::string, std::uint32_t> last_writer;
    std::vector<std::vector<std::uint32_t>> edges(passes_.size());

    for (std::uint32_t index = 0; index < passes_.size(); ++index)
    {
        const auto& pass = passes_[index];
        if (pass.name.empty())
            throw std::invalid_argument("render graph pass names must not be empty");

        for (const auto& read : pass.reads)
        {
            const auto writer = last_writer.find(read.resource);
            if (writer != last_writer.end())
                edges[writer->second].push_back(index);
        }

        for (const auto& write : pass.writes)
            last_writer[write.resource] = index;
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

    for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator)
    {
        const auto index = *iterator;
        const auto& pass = passes_[index];
        result.passes.push_back({
            .source_index = index,
            .name = pass.name,
            .queue = pass.queue,
            .kind = pass.kind
        });
    }

    return result;
}

void render_graph::clear()
{
    passes_.clear();
}

const std::vector<render_graph_pass>& render_graph::passes() const noexcept
{
    return passes_;
}

render_graph make_clear_present_graph(std::string_view target_name)
{
    std::string target(target_name);
    if (target.empty())
        target = "viewport";

    render_graph graph;
    graph.add_pass({
        .name = "clear " + target,
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::clear,
        .writes = { { .resource = target, .write = true } }
    });
    graph.add_pass({
        .name = "present " + target,
        .queue = render_queue_type::graphics,
        .kind = render_pass_kind::present,
        .reads = { { .resource = target, .write = false } }
    });
    return graph;
}

} // namespace arc::render
