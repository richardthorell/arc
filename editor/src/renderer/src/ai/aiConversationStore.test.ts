// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';

import { AI_CONVERSATION_STORE_VERSION, type AiStoredConversation } from '../../../common/aiConversationTypes';
import {
  aiProjectConversationStorageKey,
  legacyAiConversationMigrationKey,
  legacyAiConversationStorageKey,
  loadAiConversationStore,
  migrateAiConversationStoreDocument,
  saveAiConversationStore,
} from './aiConversationStore';

const projectA = '11111111-1111-1111-1111-111111111111';
const projectB = '22222222-2222-2222-2222-222222222222';

const conversation = (id: string, content = 'Hello'): AiStoredConversation => ({
  id,
  title: `Conversation ${id}`,
  createdAt: '2026-10-01T20:00:00Z',
  updatedAt: '2026-10-01T20:00:01Z',
  modelId: 'openai:gpt-5.6-sol',
  modelLabel: 'GPT-5.6 Sol',
  summary: {
    text: 'Summary',
    throughMessageId: `${id}-user`,
    updatedAt: '2026-10-01T20:00:01Z',
  },
  pinnedContext: [{ id: 'asset-1', kind: 'asset', stableId: 'asset-guid', label: 'Cabin' }],
  messages: [
    {
      id: `${id}-user`,
      role: 'user',
      content,
      createdAt: '2026-10-01T20:00:00Z',
      state: 'complete',
      contextReferences: [{ id: 'selection-1', kind: 'selection', stableId: 'entity-guid' }],
    },
    {
      id: `${id}-assistant`,
      role: 'assistant',
      content: 'Response',
      createdAt: '2026-10-01T20:00:01Z',
      state: 'complete',
      modelId: 'openai:gpt-5.6-sol',
      modelLabel: 'GPT-5.6 Sol',
      toolReferences: [
        {
          toolCallId: 'call-1',
          name: 'scene.getEntity',
          state: 'complete',
          summary: 'Completed',
          step: 0,
          arguments: { guid: 'entity-guid' },
          operation: 'scene.getEntity',
          resultContent: '{"name":"Cabin"}',
          resultTruncated: false,
          originalBytes: 16,
          startedAt: '2026-10-01T20:00:00.250Z',
          completedAt: '2026-10-01T20:00:00.500Z',
        },
      ],
      taskReferences: [
        {
          id: 'agent-step-0',
          title: 'Inspect cabin',
          state: 'completed',
          step: 0,
          toolCallIds: ['call-1'],
          startedAt: '2026-10-01T20:00:00.200Z',
          completedAt: '2026-10-01T20:00:00.500Z',
        },
      ],
    },
  ],
});

afterEach(() => localStorage.clear());

describe('AI project conversation store', () => {
  it('keeps conversation state isolated by stable project GUID', () => {
    saveAiConversationStore(projectA, [conversation('a')], {
      activeConversationId: 'a',
      selectedModelId: 'openai:gpt-5.6-sol',
    });
    saveAiConversationStore(projectB, [conversation('b')], { selectedModelId: 'anthropic:claude' });

    expect(loadAiConversationStore(projectA)).toMatchObject({
      schemaVersion: AI_CONVERSATION_STORE_VERSION,
      projectGuid: projectA,
      conversations: [{ id: 'a' }],
      uiState: { activeConversationId: 'a', selectedModelId: 'openai:gpt-5.6-sol' },
    });
    expect(loadAiConversationStore(projectB)).toMatchObject({
      projectGuid: projectB,
      conversations: [{ id: 'b' }],
      uiState: { selectedModelId: 'anthropic:claude' },
    });
    expect(aiProjectConversationStorageKey(projectA)).not.toBe(aiProjectConversationStorageKey(projectB));
  });

  it('persists context, tool/task activity, summary, and UI-state fields without credentials', () => {
    const snapshot = saveAiConversationStore(projectA, [conversation('a')], {
      activeConversationId: 'a',
      selectedModelId: 'openai:gpt-5.6-sol',
    });
    const serialized = localStorage.getItem(aiProjectConversationStorageKey(projectA)) ?? '';
    const assistant = snapshot.conversations[0].messages[1];
    const tool = assistant.toolReferences?.[0];
    const task = assistant.taskReferences?.[0];

    expect(snapshot.conversations[0].summary?.text).toBe('Summary');
    expect(snapshot.conversations[0].pinnedContext?.[0].stableId).toBe('asset-guid');
    expect(tool).toMatchObject({
      name: 'scene.getEntity',
      operation: 'scene.getEntity',
      arguments: { guid: 'entity-guid' },
      resultContent: '{"name":"Cabin"}',
      resultTruncated: false,
    });
    expect(task).toMatchObject({
      id: 'agent-step-0',
      title: 'Inspect cabin',
      state: 'completed',
      toolCallIds: ['call-1'],
    });
    expect(serialized).not.toContain('apiKey');
    expect(serialized).not.toContain('credential');
    expect(serialized).not.toContain('secret');
  });

  it('marks interrupted streaming messages, pending tools, and live tasks as interrupted when reopening a project', () => {
    const interrupted = conversation('streaming');
    interrupted.messages[1].state = 'streaming';
    interrupted.messages[1].toolReferences![0]!.state = 'pending';
    interrupted.messages[1].toolReferences![0]!.resultContent = undefined;
    interrupted.messages[1].taskReferences![0]!.state = 'in_progress';
    interrupted.messages[1].taskReferences![0]!.completedAt = undefined;
    saveAiConversationStore(projectA, [interrupted], {});

    const restored = loadAiConversationStore(projectA).conversations[0].messages[1];
    expect(restored.state).toBe('error');
    expect(restored.toolReferences?.[0]).toMatchObject({
      state: 'cancelled',
      summary: 'Interrupted when the editor closed',
    });
    expect(restored.taskReferences?.[0]).toMatchObject({
      state: 'cancelled',
      detail: 'Interrupted when the editor closed',
    });
  });

  it('migrates the old global renderer history exactly once into the active project', () => {
    const legacyConversation = conversation('legacy');
    localStorage.setItem(legacyAiConversationStorageKey, JSON.stringify([legacyConversation]));

    const migrated = loadAiConversationStore(projectA);
    expect(migrated.conversations.map((item) => item.id)).toEqual(['legacy']);
    expect(localStorage.getItem(legacyAiConversationStorageKey)).toBeNull();
    expect(localStorage.getItem(legacyAiConversationMigrationKey)).toBe(projectA);

    expect(loadAiConversationStore(projectB).conversations).toEqual([]);
  });

  it('rejects another project identity and unsupported store versions', () => {
    expect(
      migrateAiConversationStoreDocument(
        {
          schemaVersion: AI_CONVERSATION_STORE_VERSION,
          projectGuid: projectB,
          updatedAt: '2026-10-01T20:00:00Z',
          conversations: [conversation('wrong-project')],
          uiState: {},
        },
        projectA,
      ),
    ).toBeNull();

    expect(
      migrateAiConversationStoreDocument(
        {
          schemaVersion: 99,
          projectGuid: projectA,
          conversations: [],
          uiState: {},
        },
        projectA,
      ),
    ).toBeNull();
  });

  it('drops empty placeholder conversations and stale active UI references', () => {
    const raw = {
      schemaVersion: AI_CONVERSATION_STORE_VERSION,
      projectGuid: projectA,
      updatedAt: '2026-10-01T20:00:00Z',
      conversations: [
        {
          id: 'empty',
          title: 'New Chat',
          createdAt: '2026-10-01T20:00:00Z',
          updatedAt: '2026-10-01T20:00:00Z',
          messages: [],
        },
        conversation('kept'),
      ],
      uiState: { activeConversationId: 'empty', selectedModelId: 'openai:gpt-5.6-sol' },
    };
    localStorage.setItem(aiProjectConversationStorageKey(projectA), JSON.stringify(raw));

    expect(loadAiConversationStore(projectA)).toMatchObject({
      conversations: [{ id: 'kept' }],
      uiState: { selectedModelId: 'openai:gpt-5.6-sol' },
    });
    expect(loadAiConversationStore(projectA).uiState.activeConversationId).toBeUndefined();
  });
});
