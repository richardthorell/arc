#include <arc/input/input.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace arc::input
{
namespace
{

float normalize_touch_value(float value) noexcept
{
    if (!std::isfinite(value)) return 0.0f;
    return std::clamp(value, 0.0f, 1.0f);
}

} // namespace

bool input_system::submit_touch_contacts(input_device_id id, std::vector<input_touch_contact> contacts)
{
    const auto device_it = devices_.find(id.value);
    if (device_it == devices_.end()) return false;
    if (!device_it->second.connected_ && !contacts.empty()) return false;

    if (contacts.empty())
    {
        touch_contacts_.erase(id.value);
        return true;
    }

    for (input_touch_contact& contact : contacts)
    {
        contact.position[0] = normalize_touch_value(contact.position[0]);
        contact.position[1] = normalize_touch_value(contact.position[1]);

        if (!contact.pressure_available || !std::isfinite(contact.pressure))
        {
            contact.pressure = 0.0f;
            contact.pressure_available = false;
        }
        else
        {
            contact.pressure = normalize_touch_value(contact.pressure);
        }
    }

    touch_contacts_.insert_or_assign(id.value, std::move(contacts));
    return true;
}

const std::vector<input_touch_contact>& input_system::touch_contacts(input_device_id id) const noexcept
{
    static const std::vector<input_touch_contact> empty;

    const auto device_it = devices_.find(id.value);
    if (device_it == devices_.end() || !device_it->second.connected_) return empty;

    const auto contacts_it = touch_contacts_.find(id.value);
    return contacts_it == touch_contacts_.end() ? empty : contacts_it->second;
}

} // namespace arc::input
