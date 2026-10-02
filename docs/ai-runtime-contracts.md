# AI runtime contracts

ARC's built-in AI runtime uses ARC-owned request, message, tool, usage, error, and streaming-event contracts. Provider-specific OpenAI/Anthropic payloads must stop at their adapter boundary.

The canonical TypeScript contracts live in `editor/src/common/aiRuntimeTypes.ts` so renderer, preload/main-process services, and future provider adapters can share them without depending on AI Chat UI components.

The security and data-boundary rules for these contracts live in `editor/src/common/aiSecurityPolicy.ts` and are documented in `docs/ai-chat-security.md`.

## Boundary

```text
AiChatPanel / built-in agent
          |
          v
   AiRuntimeRequest
          |
          v
    AiModelProvider
          |
    +-----+------+
    |            |
 OpenAI       Anthropic
 adapter       adapter
    |            |
 provider-specific APIs
```

AI Chat owns presentation and conversation UX. Provider adapters own authentication, request translation, provider-specific streaming, and translation back into `AiRuntimeStreamEvent` values.

## Messages

`AiRuntimeMessage` supports `system`, `user`, `assistant`, and `tool` roles. Text may use a string shorthand or structured content parts. Structured content is the extension point for image/context inputs without changing provider-facing UI contracts.

```ts
{
  id: 'message-1',
  role: 'user',
  content: [{ type: 'text', text: 'Inspect the selected cabin' }]
}
```

Tool calls and tool results use ARC-owned `AiToolCall`, `AiToolDefinition`, and `AiToolResult` shapes. Tool arguments are JSON values, not provider-specific objects.

## Requests and cancellation

`AiRuntimeRequest` is the in-process runtime request and carries an optional `AbortSignal`. Cancellation is intentionally runtime-only and is not serialized.

When a request needs to cross a persistence/serialization boundary, use `serializeAiRuntimeRequest()`. It produces `AiSerializedRuntimeRequest` with `AI_RUNTIME_SCHEMA_VERSION` and excludes the signal.

This gives persisted/exported data an explicit migration boundary without forcing a serialization version into every in-memory stream call.

Provider adapters must validate ARC-owned request metadata with `assertAiRuntimeRequestSafeForProvider()` before dispatch and must propagate the runtime signal to transport cancellation. Credentials are added only at the provider transport boundary and never become runtime request fields.

## Streaming events

Provider adapters normalize their responses into this event family:

- `delta` — text delta
- `tool-call-start` — a tool invocation has started
- `tool-call-arguments-delta` — partial serialized arguments
- `tool-call` — complete validated ARC tool call
- `tool-result` — result returned by the built-in agent runtime
- `usage` — normalized token usage
- `error` — normalized runtime/provider failure
- `done` — terminal completion with an optional finish reason

The existing `delta` / `error` / `done` names remain compatible with the current Chat stream consumer while the richer tool and usage events are introduced for later agent milestones.

## Model descriptors

`AiModelDescriptor` separates ARC's stable runtime identity from provider/model identity and optional capabilities:

- `id` — unique ARC runtime model id, e.g. `openai:gpt-5.6-sol`
- `providerId` — provider family, e.g. `openai`
- `modelId` — provider model id
- `label` — display label
- `capabilities` — streaming, tool support, input modalities, and optional context/output limits

Provider discovery should populate this metadata when known. The fields remain optional on the base descriptor so existing mock/test providers can migrate incrementally without coupling UI fixtures to transport implementation details.

## Error normalization

Adapters should map provider errors into ARC error codes where possible:

`authentication`, `rate_limit`, `invalid_request`, `model_unavailable`, `context_length`, `cancelled`, `transport`, `provider`, `tool`, or `unknown`.

Use `retryable` and `retryAfterMs` when the provider supplies enough information. AI Chat should not branch on raw OpenAI or Anthropic response payloads.

Provider/service diagnostics must be redacted with the helpers in `aiSecurityPolicy.ts` and should log request summaries rather than message/context payload contents.

## Design rules

1. Provider-specific request/response types do not enter renderer UI components.
2. Built-in tools use ARC-owned definitions and results.
3. `AbortSignal` cancellation propagates through the runtime and provider adapter.
4. Serialized requests are schema-versioned and never contain runtime-only objects such as signals.
5. Model capability checks use `AiModelCapabilities` rather than provider-name heuristics.
6. Future `EditorAgentHarness` tools are projected into these contracts rather than creating a second tool protocol for AI Chat.
7. Provider credentials never become runtime message, metadata, context, tool, persistence, or diagnostic data.
8. Project/editor context must pass the AI security policy before it can be sent to an external provider.
