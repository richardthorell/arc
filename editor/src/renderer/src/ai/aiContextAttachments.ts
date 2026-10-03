import type { AiConversationMessage } from '../../../common/aiConversationTypes';
import { textContent, type AiModelCapabilities, type AiRuntimeMessage } from '../../../common/aiRuntimeTypes';

export const appendAiContextAttachments = (
  runtimeMessages: readonly AiRuntimeMessage[],
  sourceMessages: readonly AiConversationMessage[],
  capabilities: AiModelCapabilities | undefined,
): AiRuntimeMessage[] => {
  if (!capabilities?.inputModalities.includes('image')) return [...runtimeMessages];
  const sourceById = new Map(sourceMessages.map((message) => [message.id, message]));

  return runtimeMessages.map((message) => {
    const source = sourceById.get(message.id);
    const attachments = source?.contextReferences?.flatMap((reference) =>
      reference.attachment?.type === 'image' ? [reference.attachment] : [],
    );
    if (!attachments?.length) return message;
    const content = typeof message.content === 'string' ? [textContent(message.content)] : [...message.content];
    return { ...message, content: [...content, ...attachments] };
  });
};
