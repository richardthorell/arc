#pragma once

/** @file
 * @brief Game-provided native Windows packaged-client identity and registrations.
 *
 * This is deliberately separate from the legacy create_application() smoke path.
 * The host owns Win32, input, writable storage and executable discovery;
 * the shared project session owns validation, worlds and shutdown.
 */

#include <arc/project/runtime_session.h>

namespace arc::platform::windows
{

struct packaged_client_definition
{
    project::runtime_boot_expectations expected;
    project::packaged_game_registrations registrations;
    framework::application_config config;
};

/** Implemented in the game executable, with identity compiled into the binary. */
[[nodiscard]] packaged_client_definition configure_packaged_client();

} // namespace arc::platform::windows
