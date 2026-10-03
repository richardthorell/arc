// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { AiProjectContextSnapshot } from '../../../common/aiContextTypes';
import type { AiRuntimeRequest } from '../../../common/aiRuntimeTypes';
import { AiChatPanel } from './AiChatPanel';
import type { AiModelProvider } from './aiChat';

const snapshot: AiProjectContextSnapshot = {
  schemaVersion: 1,
  collectionId: 'chat-context',
  projectGuid: 'project-guid',
  capturedAt: new Date().toISOString(),
  revision: { sceneRevision: 5, eventSequence: 9 },
  sections: [
    {
      id: 'selection',
      status: 'ready',
      data: { guid: 'entity-guid', selectedGuids: ['entity-guid'], components: [{ typeId: 'arc.transform' }] },
      truncated: false,
      freshness: { capturedAt: new Date().toISOString(), ageMs: 0, cache: 'live', revision: { sceneRevision: 5 } },
      estimatedCost: { characters: 10, approximateTokens: 3 },
    },
    {
      id: 'scene',
      status: 'ready',
      data: { sceneGuid: 'scene-guid', entities: [{ guid: 'entity-guid', name: 'Hero' }] },
      truncated: false,
      freshness: { capturedAt: new Date().toISOString(), ageMs: 0, cache: 'live', revision: { sceneRevision: 5 } },
      estimatedCost: { characters: 10, approximateTokens: 3 },
    },
  ],
  estimatedCost: { characters: 20, approximateTokens: 6 },
};

const requests: AiRuntimeRequest[] = [];
const provider: AiModelProvider = {
  id: 'mock:model',
  providerId: 'mock',
  modelId: 'model',
  label: 'Mock Model',
  configured: true,
  capabilities: { streaming: true, tools: false, inputModalities: ['text'], maxContextTokens: 16_000 },
  async *stream(request) {
    requests.push(request);
    yield { type: 'done', finishReason: 'stop' };
  },
};

afterEach(() => {
  cleanup();
  requests.length = 0;
});

describe('AiChatPanel context picker', () => {
  it('opens from +, shows removable chips, and sends the reference on the user turn', async () => {
    const collect = vi.fn().mockResolvedValue(snapshot);
    render(
      <AiChatPanel
        contextSource={{ collect }}
        persistConversations={false}
        projectGuid="project-guid"
        provider={provider}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Add context' }));
    expect(await screen.findByRole('dialog', { name: 'Add context' })).toBeVisible();
    fireEvent.click(await screen.findByRole('button', { name: /Current selection/ }));

    expect(screen.getByLabelText('Attached context')).toHaveTextContent('Current selection');
    expect(screen.getByRole('button', { name: 'Remove Current selection' })).toBeVisible();

    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Inspect this entity' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start conversation' }));

    await waitFor(() => expect(requests).toHaveLength(1));
    const user = requests[0]?.messages.find((message) => message.role === 'user');
    expect(user?.content).toEqual([{ type: 'text', text: 'Inspect this entity' }]);
    expect(requests[0]?.messages.some((message) => message.role === 'system')).toBe(true);
    expect(collect).toHaveBeenCalledWith({ forceRefresh: true });
  });
});
