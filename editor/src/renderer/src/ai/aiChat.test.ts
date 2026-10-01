// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';

import {
  aiConversationStorageKey,
  conversationTitleFromPrompt,
  createAiConversation,
  createAiMessage,
  loadAiConversations,
  saveAiConversations,
} from './aiChat';

afterEach(() => localStorage.removeItem(aiConversationStorageKey));

describe('AI conversation helpers', () => {
  it('creates compact titles from the first user prompt', () => {
    expect(conversationTitleFromPrompt('  Make   the light warmer  ')).toBe('Make the light warmer');
    expect(conversationTitleFromPrompt('x'.repeat(60))).toBe(`${'x'.repeat(39)}…`);
  });

  it('persists conversations, their locked model, and interrupted stream state', () => {
    const conversation = createAiConversation();
    conversation.modelId = 'test-model';
    conversation.modelLabel = 'Test Model';
    conversation.messages.push(createAiMessage('assistant', 'partial', 'streaming'));
    saveAiConversations([conversation]);

    const restored = loadAiConversations();
    expect(restored).toHaveLength(1);
    expect(restored[0]).toMatchObject({
      id: conversation.id,
      modelId: 'test-model',
      modelLabel: 'Test Model',
    });
    expect(restored[0].messages[0]).toMatchObject({ content: 'partial', state: 'error' });
  });

  it('drops empty placeholder conversations from persisted history', () => {
    const empty = createAiConversation();
    const populated = createAiConversation();
    populated.title = 'Useful conversation';
    populated.messages.push(createAiMessage('user', 'Hello'));

    saveAiConversations([empty, populated]);

    expect(JSON.parse(localStorage.getItem(aiConversationStorageKey) ?? '[]')).toHaveLength(1);
    expect(loadAiConversations()).toEqual([
      expect.objectContaining({ id: populated.id, title: 'Useful conversation' }),
    ]);

    localStorage.setItem(aiConversationStorageKey, JSON.stringify([empty, populated]));
    expect(loadAiConversations()).toEqual([
      expect.objectContaining({ id: populated.id, title: 'Useful conversation' }),
    ]);
  });

  it('ignores malformed stored data', () => {
    localStorage.setItem(aiConversationStorageKey, JSON.stringify([{ nope: true }]));
    expect(loadAiConversations()).toEqual([]);
  });
});
