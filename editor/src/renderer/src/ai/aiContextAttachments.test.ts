import { describe, expect, it } from 'vitest';

import type { AiConversationMessage } from '../../../common/aiConversationTypes';
import type { AiModelCapabilities, AiRuntimeMessage } from '../../../common/aiRuntimeTypes';
import { appendAiContextAttachments } from './aiContextAttachments';

const message: AiConversationMessage = {
  id: 'user-1',
  role: 'user',
  content: 'What is wrong with this frame?',
  createdAt: '2026-10-03T00:00:00.000Z',
  state: 'complete',
  contextReferences: [
    {
      id: 'capture-1',
      kind: 'viewportCapture',
      label: 'Viewport capture',
      attachment: {
        type: 'image',
        mimeType: 'image/png',
        uri: 'data:image/png;base64,capture',
        alt: 'Viewport capture',
      },
    },
  ],
};

const runtime: AiRuntimeMessage = {
  id: 'user-1',
  role: 'user',
  content: 'What is wrong with this frame?',
};

const capabilities = (inputModalities: AiModelCapabilities['inputModalities']): AiModelCapabilities => ({
  streaming: true,
  tools: false,
  inputModalities,
});

describe('AI context attachments', () => {
  it('adds viewport captures to the matching runtime turn for image-capable models', () => {
    const [result] = appendAiContextAttachments([runtime], [message], capabilities(['text', 'image']));

    expect(result?.content).toEqual([
      { type: 'text', text: 'What is wrong with this frame?' },
      { type: 'image', mimeType: 'image/png', uri: 'data:image/png;base64,capture', alt: 'Viewport capture' },
    ]);
  });

  it('does not leak image data into text-only model requests', () => {
    const [result] = appendAiContextAttachments([runtime], [message], capabilities(['text']));
    expect(result?.content).toBe('What is wrong with this frame?');
  });
});
