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

  it('persists conversations and marks interrupted streams as errors on restore', () => {
    const conversation = createAiConversation();
    conversation.messages.push(createAiMessage('assistant', 'partial', 'streaming'));
    saveAiConversations([conversation]);

    const restored = loadAiConversations();
    expect(restored).toHaveLength(1);
    expect(restored[0].id).toBe(conversation.id);
    expect(restored[0].messages[0]).toMatchObject({ content: 'partial', state: 'error' });
  });

  it('ignores malformed stored data', () => {
    localStorage.setItem(aiConversationStorageKey, JSON.stringify([{ nope: true }]));
    expect(loadAiConversations()).toEqual([]);
  });
});
