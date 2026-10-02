import { ipcMain } from 'electron';

import type { AiRuntimeStreamEnvelope, AiRuntimeStreamStartRequest } from '../common/aiRuntimeIpcTypes';
import type { AiRuntimeRequest } from '../common/aiRuntimeTypes';
import { OpenAiRuntimeAdapter } from './openAiRuntimeAdapter';

export const installOpenAiRuntimeIpc = (credential: () => string | null): void => {
  const activeRequests = new Map<string, { controller: AbortController; senderId: number }>();

  ipcMain.handle('ai-runtime:start', async (ipcEvent, start: AiRuntimeStreamStartRequest) => {
    if (!start || typeof start.requestId !== 'string' || !start.requestId.trim())
      throw new Error('AI runtime request id is required');
    if (start.providerId !== 'openai')
      throw new Error(`AI provider '${String(start.providerId)}' is not supported yet`);
    if (typeof start.modelId !== 'string' || !start.modelId.trim()) throw new Error('AI model id is required');

    const apiKey = credential();
    if (!apiKey) throw new Error('OpenAI is not connected');

    const previous = activeRequests.get(start.requestId);
    if (previous) previous.controller.abort();
    const controller = new AbortController();
    activeRequests.set(start.requestId, { controller, senderId: ipcEvent.sender.id });

    const adapter = new OpenAiRuntimeAdapter(async (body, signal) =>
      fetch('https://api.openai.com/v1/responses', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal,
      }),
    );
    const request: AiRuntimeRequest = {
      conversationId: start.request.conversationId,
      messages: start.request.messages,
      ...(start.request.tools ? { tools: start.request.tools } : {}),
      ...(start.request.metadata ? { metadata: start.request.metadata } : {}),
      signal: controller.signal,
    };
    const send = (event: AiRuntimeStreamEnvelope['event']) => {
      if (ipcEvent.sender.isDestroyed()) return;
      const envelope: AiRuntimeStreamEnvelope = { requestId: start.requestId, event };
      ipcEvent.sender.send('ai-runtime:event', envelope);
    };

    try {
      for await (const event of adapter.stream(request, { modelId: start.modelId, storeResponses: false })) {
        if (controller.signal.aborted) break;
        send(event);
      }
    } finally {
      if (activeRequests.get(start.requestId)?.controller === controller) activeRequests.delete(start.requestId);
    }
  });

  ipcMain.handle('ai-runtime:cancel', (ipcEvent, requestId: string) => {
    if (typeof requestId !== 'string') return false;
    const active = activeRequests.get(requestId);
    if (!active || active.senderId !== ipcEvent.sender.id) return false;
    active.controller.abort();
    activeRequests.delete(requestId);
    return true;
  });
};
