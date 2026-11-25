#pragma once

#include <arc/framework/runtime_world.h>
#include <arc/scene/components.h>

#include <memory>

namespace arc::scene
{

enum class physics_transform_owner
{
    unbound,
    static_body,
    simulated
};

/** Exclusive, generation-bound body authority lease owned by the editor host.
 * The physics adapter must suspend simulation-driven pose writes and move the
 * body atomically before returning a lease. While held, the adapter must not
 * integrate the body or derive velocity from teleports.
 *
 * Destruction relinquishes authority at the last accepted pose, restores the
 * original motion type and clears linear/angular velocity (no throw). If the
 * body has been destroyed, release is a no-op; never retarget a recycled ID.
 */
class physics_transform_hold
{
public:
    virtual ~physics_transform_hold() = default;
    [[nodiscard]] virtual bool valid() const noexcept = 0;

    /** Synchronize the body using the entity-local pose and current hierarchy.
     * Failure must leave the previous accepted body pose unchanged.
     */
    [[nodiscard]] virtual bool set_pose(const transform_component& local) noexcept = 0;

protected:
    physics_transform_hold() = default;
};

/** Optional shared runtime service implemented by the physics ECS adapter.
 * All calls and lease destruction occur on the owning runtime thread outside
 * ECS system iteration. Runtime hosts do not depend on a particular solver.
 * Static bodies and shape scaling are deliberately outside the hold contract.
 */
class physics_transform_authority_service : public framework::runtime_service
{
public:
    static constexpr auto service_id = framework::make_runtime_service_id("arc.physics.transform_authority");

    [[nodiscard]] framework::runtime_service_id id() const noexcept final
    {
        return service_id;
    }

    [[nodiscard]] std::string_view name() const noexcept final
    {
        return "arc.physics.transform_authority";
    }

    [[nodiscard]] virtual physics_transform_owner owner(framework::runtime_world_id world,
                                                        ecs::entity entity) const noexcept = 0;

    /** Acquire and synchronize in one transaction. A null return leaves both
     * ownership and the physics pose unchanged.
     */
    [[nodiscard]] virtual std::unique_ptr<physics_transform_hold>
    acquire(framework::runtime_world_id world, ecs::entity entity, const transform_component& local) noexcept = 0;
};

} // namespace arc::scene
