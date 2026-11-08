#include "../app/src/main/cpp/android_lifecycle.h"

#include <cassert>

using namespace arc::platform::android;

int main()
{
    lifecycle_state host;
    assert(!host.can_present());
    host.apply(lifecycle_event::start);
    host.apply(lifecycle_event::resume);
    host.apply(lifecycle_event::gain_focus);
    host.apply(lifecycle_event::window_created, 1920, 1080);
    assert(host.can_present());
    assert(host.snapshot().window_generation == 1);
    host.apply(lifecycle_event::window_created, 1920, 1080);
    assert(host.snapshot().window_generation == 1); // Duplicate command
    host.apply(lifecycle_event::window_resized, 1080, 1920);
    assert(host.can_present() && host.snapshot().width == 1080);
    host.apply(lifecycle_event::pause);
    assert(!host.can_present());
    host.apply(lifecycle_event::resume);
    host.apply(lifecycle_event::window_destroyed);
    assert(!host.can_present());
    assert(host.snapshot().width == 0);
    host.apply(lifecycle_event::window_resized, 10, 10);
    assert(host.snapshot().width == 0); // Do not resurrect detached surfaces.
    host.apply(lifecycle_event::window_created, 1080, 1920);
    assert(host.snapshot().window_generation == 2);
    assert(host.can_present());
    host.apply(lifecycle_event::config_changed);
    host.apply(lifecycle_event::low_memory);
    assert(host.snapshot().window_generation == 2);
    host.apply(lifecycle_event::stop);
    assert(!host.can_present());
    host.apply(lifecycle_event::destroy);
    host.apply(lifecycle_event::window_created, 400, 400);
    assert(host.snapshot().destroy_requested);
    assert(!host.snapshot().window_available);
    return 0;
}
