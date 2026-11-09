#pragma once

#include <arc/project/distribution_finalizer.h>

namespace arc::project
{

/**
 * First Windows Shipping output adapter. Publishes a validated, self-contained
 * directory without running a cooker or depending on an installed editor/SDK.
 *
 * The caller stages and inventories the native dependency closure. The adapter
 * verifies every inventoried byte against D0 and refuses to replace an existing
 * output, then publishes with a same-parent directory rename.
 */
class windows_directory_finalizer final : public distribution_finalizer
{
public:
    [[nodiscard]] std::string_view id() const noexcept override
    {
        return "windows-directory";
    }

    [[nodiscard]] project_status validate(const distribution_finalize_request& request) const override;
    [[nodiscard]] project_status finalize(const distribution_finalize_request& request) override;
};

} // namespace arc::project
