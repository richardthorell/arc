import type { AiRuntimeStreamEnvelope, AiRuntimeStreamStartRequest } from '../../../common/aiRuntimeIpcTypes';
import {
  isTerminalAiRuntimeEvent,
  serializeAiRuntimeRequest,
  type AiRuntimeRequest,
  type AiRuntimeStreamEvent,
} from '../../../common/aiRuntimeTypes';
import { assertAiRuntimeRequestSafeForProvider } from '../../../common/aiSecurityPolicy';

export type ArcAiRuntimeBridge = {
  start(request: AiRuntimeStreamStartRequest): Promise<void>;
  cancel(requestId: string): Promise<boolean>;
  onEvent(callback: (event: AiRuntimeStreamEnvelope) => void): () => void;
};

declare global {
  interface Window {
    arcAiRuntime?: ArcAiRuntimeBridge;
  }
}

const requestId = () =>
  globalThis.crypto?.randomUUID?.() ?? `ai-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

export async function* streamOpenAiRuntime(
  modelId: string,
  request: AiRuntimeRequest,
  bridge: ArcAiRuntimeBridge | undefined = typeof window === 'undefined' ? undefined : window.arcAiRuntime,
): AsyncGenerator<AiRuntimeStreamEvent> {
  assertAiRuntimeRequestSafeForProvider(request);
  if (!bridge) {
    yield {
      type: 'error',
      code: 'provider',
      message: 'OpenAI runtime bridge is unavailable',
      retryable: false,
    };
    return;
  }
  if (request.signal?.aborted) return;

  const id = requestId();
  const queue: AiRuntimeStreamEvent[] = [];
  let terminal = false;
  let wake: (() => void) | null = null;
  const notify = () => {
    const current = wake;
    wake = null;
    current?.();
  };
  const push = (event: AiRuntimeStreamEvent) => {
    if (terminal || request.signal?.aborted) return;
    queue.push(event);
    if (isTerminalAiRuntimeEvent(event)) terminal = true;
    notify();
  };

  const unsubscribe = bridge.onEvent((envelope) => {
    if (envelope.requestId === id) push(envelope.event);
  });
  const onAbort = () => {
    terminal = true;
    void bridge.cancel(id);
    notify();
  };
  request.signal?.addEventListener('abort', onAbort, { once: true });

  const startRequest: AiRuntimeStreamStartRequest = {
    requestId: id,
    providerId: 'openai',
    modelId,
    request: serializeAiRuntimeRequest(request),
  };
  void bridge.start(startRequest).catch((error) => {
    push({
      type: 'error',
      code: 'provider',
      message: error instanceof Error ? error.message : String(error),
      retryable: false,
    });
  });

  try {
    while (!terminal || queue.length > 0) {
      if (!queue.length) {
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
        if (request.signal?.aborted) break;
      }
      const event = queue.shift();
      if (!event) continue;
      yield event;
      if (isTerminalAiRuntimeEvent(event)) break;
    }
  } finally {
    unsubscribe();
    request.signal?.removeEventListener('abort', onAbort);
    if (!terminal && !request.signal?.aborted) void bridge.cancel(id);
  }
}
