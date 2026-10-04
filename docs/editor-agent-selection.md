# Agent-driven editor selection

`EditorAgentHarness` exposes editor selection as transport-neutral UI state rather than a scene edit transaction.

- `selection.set` accepts `{ "guid": "<persistent entity guid>" }` and replaces the current editor selection.
- `selection.clear` accepts `{}` and clears the current editor selection.
- Both operations return the native `entity.selected` snapshot together with the current scene, world, and frame revisions.
- Selection uses the native `entity.select` / `entity.clearSelection` commands, so Hierarchy, Inspector, viewport highlighting, and selection events follow the same path as normal editor interaction.
- Missing or stale GUIDs fail explicitly. Selection never requires `edit.request`, never opens a history transaction, and does not mutate persistent scene content.

The operations are available to built-in tools and external gateway clients under the same harness operation names. Direct gateway routes are `/api/v1/selection/set` and `/api/v1/selection/clear`.
