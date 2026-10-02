import { ipcMain } from 'electron';

import type { AiRuntimeStreamEnvelope, AiRuntimeStreamStartRequest } from '../common/aiRuntimeIpcTypes';
import type { AiRuntimeRequest } from '../common/aiRuntimeTypes';
import { OpenAiRuntimeAdapter } from './openAiRuntimeAdapter';
import type { SettingsService } from './settingsService';

const stringSetting = (values: Record<string, unknown>, key: string): string =>
  typeof values[key] === 'string' ? String(values[key]) : '';

export const installOpenAiRuntimeIpc = (settingsService: () => SettingsService | null): void => {
  const activeRequests = new Map<string, { controller: AbortController; senderId: number }>();

  ipcMain.handle('ai-runtime:start', async (ipcEvent, start: AiRuntimeStreamStartRequest) => {
    if (!start || typeof start.requestId !== 'string' || !start.requestId.trim())
      throw new Error('AI runtime request id is required');
    if (start.providerId !== 'openai') throw new Error(`AI provider '${String(start.providerId)}' is not supported yet`);
    if (typeof start.modelId !== 'string' || !start.modelId.trim()) throw new Error('AI model id is required');

    const service = settingsService();
    if (!service) throw new Error('Editor settings are unavailable');
    const credential = service.providerCredential('openai');
    if (!credential) throw new Error('OpenAI is not connected');

    const snapshot = service.snapshot();
    const values = snapshot.values;
    const organizationId = stringSetting(values, 'ai.openai.organizationId').trim();
    const projectId = stringSetting(values, 'ai.openai.projectId').trim();
    const reasoningEffort = stringSetting(values, 'ai.openai.reasoningEffort');
    const storeResponses = values['ai.openai.storeResponses'] === true;

    const previous = activeRequests.get(start.requestId);
    if (previous) previous.controller.abort();
    const controller = new AbortController();
    activeRequests.set(start.requestId, { controller, senderId: ipcEvent.sender.id });

    const transport = async (body: Record<string, unknown>, signal?: AbortSignal): Promise<Response> => {
      const headers = new Headers({ 'Content-Type': 'application/json' });
      headers.set('Authorization', `Bearer ${credential}`);
      if (organizationId) headers.set('OpenAI-Organization', organizationId);
      if (projectId) headers.set('OpenAI-Project', projectId);
      return fetch('https://api.openai.com/v1/responses', {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal,
      });
    };
    const adapter = new OpenAiRuntimeAdapter(transport);
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
      for await (const event of adapter.stream(request, {
        modelId: start.modelId,
        ...(reasoningEffort ? { reasoningEffort } : {}),
        storeResponses,
      })) {
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
}
