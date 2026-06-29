#include <arc/render/handles.h>

namespace arc::render
{

resource_handle handle_pool::allocate()
{
    if (!free_list_.empty())
    {
        const std::uint32_t index = free_list_.back();
        free_list_.pop_back();
        slot& value = slots_[index];
        value.alive = true;
        ++live_count_;
        return { .index = index, .generation = value.generation };
    }

    const auto index = static_cast<std::uint32_t>(slots_.size());
    slots_.push_back({ .generation = 1, .alive = true });
    ++live_count_;
    return { .index = index, .generation = 1 };
}

bool handle_pool::release(resource_handle handle)
{
    if (!alive(handle))
        return false;

    slot& value = slots_[handle.index];
    value.alive = false;
    ++value.generation;
    if (value.generation == 0)
        value.generation = 1;
    free_list_.push_back(handle.index);
    --live_count_;
    return true;
}

bool handle_pool::alive(resource_handle handle) const
{
    return handle.valid() &&
        handle.index < slots_.size() &&
        slots_[handle.index].alive &&
        slots_[handle.index].generation == handle.generation;
}

std::uint32_t handle_pool::live_count() const noexcept
{
    return live_count_;
}

} // namespace arc::render
