import { describe, expect, it } from 'vitest';

import { conversationTitleFromPrompt, createAiConversation, createAiMessage } from './aiChat';

describe('AI conversation helpers', () => {
  it('creates compact titles from the first user prompt', () => {
    expect(conversationTitleFromPrompt('  Make   the light warmer  ')).toBe('Make the light warmer');
    expect(conversationTitleFromPrompt('x'.repeat(60))).toBe(`${'x'.repeat(39)}…`);
  });

  it('creates conversations and messages with persistence-ready metadata', () => {
    const conversation = createAiConversation();
    const message = createAiMessage('assistant', 'Hello');

    expect(conversation).toMatchObject({ title: 'New Chat', messages: [] });
    expect(conversation.id).toBeTruthy();
    expect(conversation.createdAt).toBeTruthy();
    expect(message).toMatchObject({ role: 'assistant', content: 'Hello', state: 'complete' });
    expect(message.id).toBeTruthy();
    expect(message.createdAt).toBeTruthy();
  });
});
