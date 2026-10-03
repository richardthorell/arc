import { ipcMain, type WebContents } from 'electron';

import type { BuiltInAgentInvokeRequest } from '../common/builtInAgentTypes';
import { BuiltInAgentAdapter } from './builtInAgentAdapter';

const capabilitiesChannel = 'ai-runtime:agent-capabilities';
const invokeChannel = 'ai-runtime:agent-invoke';
const subscribeChannel = 'ai-runtime:agent-subscribe';
const unsubscribeChannel = 'ai-runtime:agent-unsubscribe';
const eventChannel = 'ai-runtime:agent-event';

const invokeRequest = (value: unknown): BuiltInAgentInvokeRequest => {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Built-in agent invocation request is invalid');
  const request = value as Record<string, unknown>;
  if (typeof request.method !== 'string' || request.method.trim() === '')
    throw new Error('Built-in agent method is required');
  return {
    method: request.method,
    ...(Object.hasOwn(request, 'params') ? { params: request.params } : {}),
  };
};

export const installBuiltInAgentIpc = (adapter: BuiltInAgentAdapter): (() => void) => {
  const subscriptions = new Map<number, () => void>();

  const unsubscribe = (senderId: number): void => {
    subscriptions.get(senderId)?.();
    subscriptions.delete(senderId);
  };

  const subscribe = (sender: WebContents): void => {
    if (subscriptions.has(sender.id)) return;
    const unsubscribeHarness = adapter.onEvent((event) => {
      if (!sender.isDestroyed()) sender.send(eventChannel, event);
    });
    const cleanup = () => unsubscribe(sender.id);
    sender.once('destroyed', cleanup);
    subscriptions.set(sender.id, () => {
      sender.removeListener('destroyed', cleanup);
      unsubscribeHarness();
    });
  };

  ipcMain.handle(capabilitiesChannel, () => adapter.capabilities());
  ipcMain.handle(invokeChannel, (_event, rawRequest: unknown) => {
    const request = invokeRequest(rawRequest);
    return adapter.invoke(request.method, request.params);
  });
  ipcMain.handle(subscribeChannel, (event) => {
    subscribe(event.sender);
    return true;
  });
  ipcMain.handle(unsubscribeChannel, (event) => {
    unsubscribe(event.sender.id);
    return true;
  });

  return () => {
    ipcMain.removeHandler(capabilitiesChannel);
    ipcMain.removeHandler(invokeChannel);
    ipcMain.removeHandler(subscribeChannel);
    ipcMain.removeHandler(unsubscribeChannel);
    for (const senderId of [...subscriptions.keys()]) unsubscribe(senderId);
  };
};
