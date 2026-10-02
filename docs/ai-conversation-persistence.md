# AI conversation persistence

ARC AI Chat stores conversation history as user-private editor state scoped by the active project's stable project GUID.

## Storage boundary

Conversation history is intentionally **not** written into the project repository. The current backing store is the Electron renderer profile storage, namespaced by project GUID:

```text
arc.ai.projects.<project-guid>.conversations.v1
```

The renderer obtains the active GUID from `ArcProjectDescriptor.guid` through the normal project snapshot API. `AiGatewayPanel` remounts the production `AiChatPanel` when that GUID changes, so switching or closing projects cannot retain another project's in-memory conversation state.

The backing store is isolated behind `aiConversationStore.ts`; AI Chat UI code consumes the store contract rather than constructing storage keys itself. A later disk/IPC-backed implementation can replace the backing mechanism without changing the persisted schema or conversation UI model.

## Versioned schema

`editor/src/common/aiConversationTypes.ts` defines `AI_CONVERSATION_STORE_VERSION` and the persisted shapes shared by the chat runtime and future context/tool work.

A snapshot contains:

- project GUID
- schema version and update timestamp
- conversations and messages
- model identity per conversation and response
- optional conversation summary
- optional pinned and per-message context references
- optional tool-call references
- UI state such as the active conversation and selected model

Credentials and provider secrets are deliberately absent from this schema.

Interrupted `streaming` messages are restored as `error` messages after restart because an in-flight provider stream cannot be resumed safely from persisted UI state.

## Migration

The pre-project-scoping implementation stored one global bare conversation array under:

```text
arc.ai.conversations.v1
```

On the first project-scoped load, ARC imports that legacy history exactly once into the currently active project, removes the global source, and records a migration marker. Since the old format had no project identity, assigning it once is the only deterministic migration that prevents the same legacy history from appearing in multiple projects.

Future schema migrations should be added to `migrateAiConversationStoreDocument()` and preserve the rule that a stored snapshot is accepted only for its matching project GUID.

## Lifecycle rules

1. Use `ArcProjectDescriptor.guid`, never project display name or path, as the persistence identity.
2. Production chat persistence is enabled only while a project GUID is available.
3. UI Lab/tests may provide fixture conversations with persistence disabled.
4. Closing/switching projects changes the chat component key and discards ephemeral prompt/stream state.
5. Empty placeholder conversations are not persisted.
6. A persisted active-conversation reference is discarded if its conversation no longer exists.
7. Provider credentials remain in the existing secure provider service and never enter conversation storage.
