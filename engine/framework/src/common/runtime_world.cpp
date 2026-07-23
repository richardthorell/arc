#include <arc/framework/runtime_world.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace arc
{
namespace
{

constexpr std::size_t estimated_entity_snapshot_bytes = 256;

constexpr int role_order(runtime_world_role role) noexcept
{
    switch (role)
    {
    case runtime_world_role::server: return 0;
    case runtime_world_role::client: return 1;
    case runtime_world_role::editor_preview: return 2;
    }
    return 3;
}

const char* placeholder_name(ecs::system_phase phase) noexcept
{
    switch (phase)
    {
    case ecs::system_phase::input: return "ARC Input Boundary";
    case ecs::system_phase::network_receive: return "ARC Network Receive TODO";
    case ecs::system_phase::gameplay_commands: return "ARC Gameplay Commands TODO";
    case ecs::system_phase::movement: return "ARC Movement TODO";
    case ecs::system_phase::physics: return "ARC Physics TODO";
    case ecs::system_phase::abilities: return "ARC Abilities TODO";
    case ecs::system_phase::ai: return "ARC AI TODO";
    case ecs::system_phase::replication: return "ARC Replication TODO";
    case ecs::system_phase::presentation_extraction: return "ARC Presentation Extraction Boundary";
    }
    return "ARC Runtime Placeholder";
}

class execution_scope
{
public:
    explicit execution_scope(bool& executing) noexcept
        : executing_(&executing)
    {
        executing = true;
    }

    ~execution_scope()
    {
        *executing_ = false;
    }

private:
    bool* executing_;
};

} // namespace

runtime_world::runtime_world(
    memory_system& memory,
    runtime_world_id id,
    runtime_world_descriptor descriptor)
    : id_(id)
    , descriptor_(std::move(descriptor))
    , owned_entities_(memory, id.value, descriptor_.memory)
    , entities_(&owned_entities_)
{
    if (descriptor_.name.empty())
        descriptor_.name = "World " + std::to_string(id.value);
    if (descriptor_.seed == 0)
        descriptor_.seed = ecs::splitmix64(id.value);
    if (descriptor_.install_placeholder_systems)
        install_placeholder_systems();
}

void runtime_world::install_placeholder_systems()
{
    constexpr std::array phases{
        ecs::system_phase::input,
        ecs::system_phase::network_receive,
        ecs::system_phase::gameplay_commands,
        ecs::system_phase::movement,
        ecs::system_phase::physics,
        ecs::system_phase::abilities,
        ecs::system_phase::ai,
        ecs::system_phase::replication,
        ecs::system_phase::presentation_extraction
    };
    for (const ecs::system_phase phase : phases)
    {
        systems_.add({
            .name = placeholder_name(phase),
            .phase = phase,
            .execute = [](ecs::system_context&) {
                // TODO: Replace this explicit scheduling boundary when the subsystem lands.
            }
        });
    }
}

void runtime_world::start()
{
    if (state_ == runtime_world_state::running || state_ == runtime_world_state::paused)
        return;
    state_ = runtime_world_state::starting;
    fault_message_.clear();
    const std::vector<ecs::system_schedule_error> schedule_errors = systems_.freeze();
    if (!schedule_errors.empty())
    {
        fault(schedule_errors.front().message);
        throw std::invalid_argument(
            "runtime world '" + descriptor_.name +
            "' has an invalid system schedule: " + fault_message_);
    }
    state_ = runtime_world_state::running;
}

void runtime_world::stop() noexcept
{
    if (state_ == runtime_world_state::stopped)
        return;
    state_ = runtime_world_state::stopping;
    state_ = runtime_world_state::stopped;
}

void runtime_world::pause() noexcept
{
    if (state_ == runtime_world_state::running)
        state_ = runtime_world_state::paused;
}

void runtime_world::resume() noexcept
{
    if (state_ == runtime_world_state::paused)
        state_ = runtime_world_state::running;
}

void runtime_world::clear_fault() noexcept
{
    if (state_ == runtime_world_state::faulted)
    {
        fault_message_.clear();
        state_ = runtime_world_state::paused;
    }
}

bool runtime_world::attach_entities(ecs::world& entities) noexcept
{
    if (state_ == runtime_world_state::starting ||
        state_ == runtime_world_state::running ||
        state_ == runtime_world_state::stopping)
        return false;
    entities_ = &entities;
    ++epoch_;
    return true;
}

void runtime_world::fault(std::string message)
{
    fault_message_ = std::move(message);
    state_ = runtime_world_state::faulted;
}

runtime_world_run_result runtime_world::run_phase(
    job_system& jobs,
    ecs::system_phase phase,
    const ecs::system_execution_info& execution)
{
    runtime_world_run_result result;
    ecs::system_run_result run = systems_.run_phase(*entities_, jobs, phase, execution);
    result.systems_executed = run.systems_executed;
    result.errors = std::move(run.errors);
    for (const ecs::command_error& error : run.commands.errors)
        result.errors.push_back({ {}, error.message });
    if (!result.succeeded())
        fault(result.errors.front().message);
    return result;
}

runtime_world_run_result runtime_world::run_fixed(
    job_system& jobs,
    const simulation_tick& tick,
    runtime_service_provider* services,
    const simulation_input_snapshot* input,
    tick_arena& tick_memory,
    frame_arena& frame_memory,
    std::uint64_t process_seed)
{
    runtime_world_run_result aggregate;
    if (state_ != runtime_world_state::running)
        return aggregate;

    ecs::system_execution_info execution{};
    execution.tick = tick.id;
    execution.world_role = role();
    execution.world_id = id_.value;
    execution.process_seed = process_seed ^ descriptor_.seed;
    execution.delta_seconds = static_cast<float>(tick.delta_seconds);
    execution.fixed_delta_seconds = static_cast<float>(tick.delta_seconds);
    execution.frame_delta_seconds = static_cast<float>(tick.delta_seconds);
    execution.input = input;
    execution.services = services;
    execution.tick_memory = &tick_memory;
    execution.frame_memory = &frame_memory;

    for (const ecs::system_phase phase : ecs::fixed_system_phases)
    {
        runtime_world_run_result phase_result = run_phase(jobs, phase, execution);
        aggregate.systems_executed += phase_result.systems_executed;
        aggregate.errors.insert(
            aggregate.errors.end(),
            std::make_move_iterator(phase_result.errors.begin()),
            std::make_move_iterator(phase_result.errors.end()));
        if (!phase_result.succeeded())
            return aggregate;
    }
    last_completed_tick_ = tick.id;
    return aggregate;
}

runtime_world_run_result runtime_world::run_presentation(
    job_system& jobs,
    const simulation_tick& tick,
    float frame_delta_seconds,
    float interpolation_alpha,
    runtime_service_provider* services,
    const simulation_input_snapshot* input,
    tick_arena& tick_memory,
    frame_arena& frame_memory,
    std::uint64_t process_seed)
{
    if (state_ != runtime_world_state::running && state_ != runtime_world_state::paused)
        return {};
    if (!descriptor_.presentation_enabled)
        return {};

    ecs::system_execution_info execution{};
    execution.tick = tick.id;
    execution.world_role = role();
    execution.world_id = id_.value;
    execution.process_seed = process_seed ^ descriptor_.seed;
    execution.delta_seconds = static_cast<float>(tick.delta_seconds);
    execution.fixed_delta_seconds = static_cast<float>(tick.delta_seconds);
    execution.frame_delta_seconds = frame_delta_seconds;
    execution.interpolation_alpha = interpolation_alpha;
    execution.presentation = true;
    execution.input = input;
    execution.services = services;
    execution.tick_memory = &tick_memory;
    execution.frame_memory = &frame_memory;
    return run_phase(jobs, ecs::system_phase::presentation_extraction, execution);
}

runtime_world_manager::runtime_world_manager(memory_system& memory)
    : memory_(&memory)
{
}

runtime_world& runtime_world_manager::create(runtime_world_descriptor descriptor)
{
    if (executing_)
        throw std::logic_error("runtime worlds may only be created at frame boundaries");
    const runtime_world_id id{ next_world_id_++ };
    auto value = std::make_unique<runtime_world>(*memory_, id, std::move(descriptor));
    runtime_world& result = *value;
    worlds_.push_back(std::move(value));
    rebuild_execution_order();
    return result;
}

bool runtime_world_manager::destroy(runtime_world_id id)
{
    if (executing_)
        return false;
    const auto found = std::find_if(worlds_.begin(), worlds_.end(), [id](const auto& value) {
        return value->id() == id;
    });
    if (found == worlds_.end())
        return false;
    (*found)->stop();
    worlds_.erase(found);
    rebuild_execution_order();
    return true;
}

runtime_world* runtime_world_manager::find(runtime_world_id id) noexcept
{
    const auto found = std::find_if(worlds_.begin(), worlds_.end(), [id](const auto& value) {
        return value->id() == id;
    });
    return found == worlds_.end() ? nullptr : found->get();
}

const runtime_world* runtime_world_manager::find(runtime_world_id id) const noexcept
{
    const auto found = std::find_if(worlds_.begin(), worlds_.end(), [id](const auto& value) {
        return value->id() == id;
    });
    return found == worlds_.end() ? nullptr : found->get();
}

const std::vector<runtime_world*>& runtime_world_manager::execution_order() const noexcept
{
    return execution_order_;
}

void runtime_world_manager::rebuild_execution_order()
{
    execution_order_.clear();
    execution_order_.reserve(worlds_.size());
    for (const auto& world : worlds_)
        execution_order_.push_back(world.get());
    std::sort(
        execution_order_.begin(),
        execution_order_.end(),
        [](const runtime_world* lhs, const runtime_world* rhs) {
            const int lhs_role = role_order(lhs->role());
            const int rhs_role = role_order(rhs->role());
            return lhs_role != rhs_role ? lhs_role < rhs_role : lhs->id() < rhs->id();
        });
}

std::vector<runtime_world_id> runtime_world_manager::ordered_worlds() const
{
    std::vector<runtime_world_id> result;
    result.reserve(execution_order_.size());
    for (const runtime_world* world : execution_order_)
        result.push_back(world->id());
    return result;
}

runtime_world_run_result runtime_world_manager::run_fixed(
    job_system& jobs,
    const simulation_tick& tick,
    runtime_service_provider* services,
    const simulation_input_snapshot* input,
    tick_arena& tick_memory,
    frame_arena& frame_memory,
    std::uint64_t process_seed)
{
    runtime_world_run_result aggregate;
    execution_scope scope(executing_);
    for (runtime_world* world : execution_order())
    {
        runtime_world_run_result result = world->run_fixed(
            jobs, tick, services, input, tick_memory, frame_memory, process_seed);
        aggregate.systems_executed += result.systems_executed;
        aggregate.errors.insert(
            aggregate.errors.end(),
            std::make_move_iterator(result.errors.begin()),
            std::make_move_iterator(result.errors.end()));
    }
    return aggregate;
}

runtime_world_run_result runtime_world_manager::run_presentation(
    job_system& jobs,
    const simulation_tick& tick,
    float frame_delta_seconds,
    float interpolation_alpha,
    runtime_service_provider* services,
    const simulation_input_snapshot* input,
    tick_arena& tick_memory,
    frame_arena& frame_memory,
    std::uint64_t process_seed)
{
    runtime_world_run_result aggregate;
    execution_scope scope(executing_);
    for (runtime_world* world : execution_order())
    {
        runtime_world_run_result result = world->run_presentation(
            jobs,
            tick,
            frame_delta_seconds,
            interpolation_alpha,
            services,
            input,
            tick_memory,
            frame_memory,
            process_seed);
        aggregate.systems_executed += result.systems_executed;
        aggregate.errors.insert(
            aggregate.errors.end(),
            std::make_move_iterator(result.errors.begin()),
            std::make_move_iterator(result.errors.end()));
    }
    return aggregate;
}

void runtime_world_manager::start_all()
{
    for (runtime_world* world : execution_order())
        world->start();
}

void runtime_world_manager::stop_all() noexcept
{
    for (auto iterator = worlds_.rbegin(); iterator != worlds_.rend(); ++iterator)
        (*iterator)->stop();
}

void runtime_world_manager::pause_all() noexcept
{
    for (auto& world : worlds_)
        world->pause();
}

void runtime_world_manager::resume_all() noexcept
{
    for (auto& world : worlds_)
        world->resume();
}

void runtime_world_manager::set_snapshot_budget(std::size_t bytes) noexcept
{
    snapshot_budget_bytes_ = bytes;
    trim_snapshots();
}

world_snapshot_result runtime_world_manager::capture_snapshot(
    runtime_world_id world_id,
    simulation_tick_id tick,
    double simulation_time_seconds,
    std::string label,
    runtime_service_registry* services)
{
    if (executing_)
        return { false, {}, "snapshots may only be captured at a phase boundary" };
    if (snapshot_budget_bytes_ == 0)
        return { false, {}, "runtime snapshots are disabled" };
    runtime_world* world = find(world_id);
    if (!world)
        return { false, {}, "runtime world does not exist" };

    std::vector<runtime_service_snapshot> service_snapshots;
    std::string service_error;
    if (services && !services->capture_deterministic_state(
        world_id.value, service_snapshots, service_error))
        return { false, {}, std::move(service_error) };
    std::size_t service_bytes{};
    for (const runtime_service_snapshot& snapshot : service_snapshots)
        service_bytes += snapshot.bytes.size();

    world_snapshot_metadata metadata{
        .id = { next_snapshot_id_++ },
        .world = world_id,
        .tick = tick,
        .simulation_time_seconds = simulation_time_seconds,
        .world_epoch = world->epoch_,
        .estimated_bytes = std::max<std::size_t>(
            estimated_entity_snapshot_bytes,
            world->entities_->live_count() * estimated_entity_snapshot_bytes) + service_bytes,
        .label = std::move(label)
    };
    if (snapshot_budget_bytes_ != 0 && metadata.estimated_bytes > snapshot_budget_bytes_)
        return { false, {}, "snapshot exceeds the configured snapshot budget" };

    stored_snapshot snapshot{
        .metadata = metadata,
        .descriptor = world->descriptor_,
        .entities = *world->entities_,
        .partition = world->partition_,
        .services = std::move(service_snapshots),
        .last_completed_tick = world->last_completed_tick_
    };
    snapshot_bytes_ += metadata.estimated_bytes;
    snapshots_.push_back(std::move(snapshot));
    trim_snapshots();
    return { true, metadata, {} };
}

world_snapshot_result runtime_world_manager::restore_snapshot(
    world_snapshot_id id,
    runtime_service_registry* services)
{
    if (executing_)
        return { false, {}, "snapshots may only be restored at a phase boundary" };
    const auto found = std::find_if(snapshots_.begin(), snapshots_.end(), [id](const stored_snapshot& value) {
        return value.metadata.id == id;
    });
    if (found == snapshots_.end())
        return { false, {}, "snapshot does not exist" };

    runtime_world* world = find(found->metadata.world);
    if (!world)
        return { false, found->metadata, "snapshot target world does not exist" };

    std::string service_error;
    if (!found->services.empty() &&
        (!services || !services->validate_deterministic_state(
            world->id_.value, found->services, service_error)))
        return {
            false,
            found->metadata,
            service_error.empty()
                ? "snapshot deterministic runtime services are unavailable"
                : std::move(service_error)
        };

    try
    {
        ecs::world staged = found->entities;
        ecs::world_partition staged_partition = found->partition;
        world->entities_->swap(staged);
        world->partition_ = std::move(staged_partition);
        world->descriptor_.seed = found->descriptor.seed;
        world->last_completed_tick_ = found->last_completed_tick;
        if (services)
            services->restore_deterministic_state(world->id_.value, found->services);
        ++world->epoch_;
        world->fault_message_.clear();
        if (world->state_ == runtime_world_state::faulted)
            world->state_ = runtime_world_state::paused;
        world_snapshot_metadata restored = found->metadata;
        restored.world_epoch = world->epoch_;
        return { true, std::move(restored), {} };
    }
    catch (const std::exception& error)
    {
        return { false, found->metadata, error.what() };
    }
}

std::vector<world_snapshot_metadata> runtime_world_manager::snapshots() const
{
    std::vector<world_snapshot_metadata> result;
    result.reserve(snapshots_.size());
    for (const stored_snapshot& snapshot : snapshots_)
        result.push_back(snapshot.metadata);
    return result;
}

void runtime_world_manager::trim_snapshots()
{
    if (snapshot_budget_bytes_ == 0)
    {
        snapshots_.clear();
        snapshot_bytes_ = 0;
        return;
    }
    while (!snapshots_.empty() && snapshot_bytes_ > snapshot_budget_bytes_)
    {
        snapshot_bytes_ -= snapshots_.front().metadata.estimated_bytes;
        snapshots_.pop_front();
    }
}

} // namespace arc
