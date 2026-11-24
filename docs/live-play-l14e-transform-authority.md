# Live Play L1.4e — explicit native transform authority (Windows, first slice)

Tracking: [#1199](https://github.com/richardthorell/arc/issues/1199). Follows merged L1.4c (#1245), L1.4d (#1248), and verified native unsaved baseline (#1247).

## Policy

- **Default:** external Play belongs to the simulation. An entity with an enabled native `scene::flow_component` is treated conservatively as gameplay-owned. Normal `entity.property.set` transform `update`/`commit` commands **reject with `permission_denied`** unless an explicit hold has already been granted. Existing unaffiliated movable authored transforms preserve L1.4a–d behavior. Static/stationary mobility still rejects edits.
- **Acquire:** an authenticated `entity.property.set` for `arc.transform` schema v1 with phase `begin` acquires a **session-local, bounded (64-entity)** hold for that exact authored GUID and native ECS entity generation. It applies the requested finite typed transform value at the existing safe-frame edit boundary and ACKs an applied revision. Other light/component schemas cannot acquire holds.
- **Edit while held:** subsequent transform `update` and reliably retried `commit` commands refresh the held full transform. Revision checks, exact-value duplicate ACKs, authoring GUID binding and maximum request sizes are reused from L1.4c. No arbitrary component changes are permitted.
- **Presentation:** an optional platform-neutral runtime hook runs after all fixed simulation ticks and **before** presentation systems and application rendering. Windows Development Live Play uses it to reassert the held transform if gameplay has overwritten it; the default hook is empty in ordinary runtimes. This prevents a continuous conflict at the visible-frame boundary without mutating ECS on a socket callback.
- **Release:** `cancel` releases a held authored transform and ACKs without restoring pre-drag authoring data. Runtime gameplay resumes from the last current ECS transform on subsequent ticks. A duplicate committed command cannot recreate the hold.
- **Revocation:** loss of the verified baseline, disconnection, Stop or host shutdown clears holds and command/identity history. The frame hook also drops holds when the world stops, entity generation dies, GUID changes, transform disappears, or mobility becomes non-movable. Deleted/recreated ECS entities sharing the same authored GUID are rejected until a fresh baseline/connection, never silently retargeted.

## Explicit non-goals and remaining L1.4e work

- **No UI affordance yet.** The native `begin`/`cancel` phases are implemented; the next editor-side slice must expose deliberate override/hold/release controls (including tracking outstanding hold ACKs), without automatically overriding gameplay when any Inspector property is edited.
- **Physics integration pending.** ARC's current backend-neutral physics contract defines body motion, but this runtime scene does not yet expose a shared ECS rigid-body ownership/velocity adapter. A future adapter must register dynamic-body ownership, suspend/synchronize body motion during hold, and define velocity/wake/impulse handling on release. Merely reasserting ECS transforms must **not** be mistaken for complete rigid-body ownership transfer.
- **No unconditional authority for static/stationary objects, no create/delete/reparent transactions, no structural scene recook, no Shipping behavior, no multiplayer replication.**
- **Acceptance still open:** full Windows sample with actual gameplay/physics ownership, interactive override UI, visible cube/light changes and measured edit-to-visible-frame latency (target <100 ms); then Android parity. See #1199 and #1142.

## Test coverage

`windows_live_play_edits_tests` includes: implicit edit rejection on Flow-owned entities; explicit begin + idempotent ACK; simulated competing runtime transform write then pre-presentation pose restoration; final commit retention; explicit cancel and gameplay resumption; baseline-revoked hold clearing; identity generation invalidation without touching replacement entities.

Development-only control-plane property commands remain independent of C9's authenticated cooked content data plane; no project save, secrets, or source-file access are introduced.
