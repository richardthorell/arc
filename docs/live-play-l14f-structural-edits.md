# L1.4f.1 — Bounded structural edit foundation

The Development bridge accepts the existing v1 `entity.create` and `entity.delete` messages through the authenticated `entity.edit` capability, active baseline gate, and bounded edit mailbox. Create supports only the empty root archetype (`parent: null`); delete only entities created by this bridge with no children, active hold, Flow or physics binding. The runtime owns identity and mutation on the frame thread. GUID tombstones prevent reuse and duplicate commands replay original ACKs only in the same baseline epoch. There is a 256-identity lifetime cap per attached runtime.

The editor control server and session expose trusted main-process send methods; automatic authoring structure capture, rich archetypes, baseline entity deletion, reparenting and structural Undo/Redo are follow-up work. Existing editor snapshots still fail closed on structural changes. This is a native foundation, not end-to-end structural editing. Refs #1199.
