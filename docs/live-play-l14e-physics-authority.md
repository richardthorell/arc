# L1.4e.3 — Physics transform authority

Live Play now calls the optional shared scene runtime service
`arc.physics.transform_authority` when a physics adapter declares a body owner.
The service is independent of the Windows transport and of a particular solver.

- Unbound entities retain the existing gameplay/Flow behavior.
- Static bodies reject edits. Simulated bodies require an explicit hold.
- Acquisition is transactional: the adapter suspends physics pose writes and
  synchronizes the body before ECS is mutated and the command is acknowledged.
- During the hold, position/rotation updates use the last held pose and are
  synchronized with the body before ECS changes. Shape scaling is unsupported.
- Cancel and all existing hold cleanup paths destroy the generation-bound
  lease. The adapter must restore original motion at the last accepted pose,
  zero linear and angular velocity, and do nothing to a destroyed body.

The zero-velocity policy intentionally avoids launching a body with an old
velocity or teleport-derived impulse. The adapter is responsible for mapping
local transforms to physics world space and respecting its hierarchy constraints.

## Scope and limitations

ARC currently has a physics foundation and private Jolt dependency, but no
production physics world/body creation, ECS-to-body binding, or simulation
adapter. This change is the real runtime integration seam and contract, tested
with an in-memory adapter; it does not claim production Jolt simulation.
A follow-up must implement that adapter and verify contacts, motion restoration,
velocity clearing and hierarchy behavior against real bodies. Issue #1199 stays
open until that and end-to-end Windows latency acceptance are complete.
