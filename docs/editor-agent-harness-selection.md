# Editor agent selection

ARC agent selection is editor/UI state, not a persistent scene edit.

`selection.set` accepts a persistent entity GUID and replaces the current selection. `selection.clear` clears it. Both operations use the native editor selection commands and return the authoritative selected-entity snapshot together with scene, world, and frame revisions.

Because the native selection path is shared with normal editor interaction, Hierarchy, Inspector, viewport highlighting, and `entity.selected` events remain consistent regardless of whether selection came from a person, the built-in agent, or an external AI Gateway client.

Selection operations do not require edit approval and never open an edit/history transaction. Missing or stale GUIDs fail explicitly.
