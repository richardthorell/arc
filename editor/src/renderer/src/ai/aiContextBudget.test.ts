import { describe, expect, it } from 'vitest';

import {
  AI_PROJECT_CONTEXT_SCHEMA_VERSION,
  type AiContextJsonValue,
  type AiContextRevision,
  type AiContextSection,
  type AiProjectContextSnapshot,
} from '../../../common/aiContextTypes';
import type {
  AiConversationContextReference,
  AiConversationMessage,
  AiStoredConversation,
} from '../../../common/aiConversationTypes';
import { prepareAiContextBudget, type AiProjectContextSource } from './aiContextBudget';

const message = (
  id: string,
  role: AiConversationMessage['role'],
  content: string,
  contextReferences?: AiConversationContextReference[],
): AiConversationMessage => ({
  id,
  role,
  content,
  createdAt: `2026-10-03T00:${id.padStart(2, '0')}:00.000Z`,
  state: 'complete',
  ...(contextReferences ? { contextReferences } : {}),
});

const conversation = (
  messages: AiConversationMessage[],
  options: Partial<Pick<AiStoredConversation, 'summary' | 'pinnedContext'>> = {},
): AiStoredConversation => ({
  id: 'conversation-1',
  title: 'Context budget test',
  createdAt: '2026-10-03T00:00:00.000Z',
  updatedAt: '2026-10-03T01:00:00.000Z',
  messages,
  ...options,
});

const section = (
  id: AiContextSection['id'],
  data: AiContextJsonValue,
  revision: AiContextRevision,
  ageMs = 0,
): AiContextSection => ({
  id,
  status: 'ready',
  data,
  truncated: false,
  freshness: {
    capturedAt: '2026-10-03T01:00:00.000Z',
    ageMs,
    cache: ageMs ? 'cached' : 'live',
    revision,
  },
  estimatedCost: {
    characters: JSON.stringify(data).length,
    approximateTokens: Math.ceil(JSON.stringify(data).length / 4),
  },
});

const contextSnapshot = (
  revision: AiContextRevision,
  sceneData: AiContextJsonValue = { entities: [] },
  projectGuid = 'project-a',
  ageMs = 0,
): AiProjectContextSnapshot => {
  const sections = [
    section('project', { guid: projectGuid, name: 'Budget Project' }, revision, ageMs),
    section('scene', sceneData, revision, ageMs),
    section('selection', { selectedGuids: [] }, revision, ageMs),
  ];
  return {
    schemaVersion: AI_PROJECT_CONTEXT_SCHEMA_VERSION,
    collectionId: `collection-${revision.sceneRevision ?? 0}`,
    projectGuid,
    capturedAt: '2026-10-03T01:00:00.000Z',
    revision,
    sections,
    estimatedCost: {
      characters: sections.reduce((total, candidate) => total + candidate.estimatedCost.characters, 0),
      approximateTokens: sections.reduce((total, candidate) => total + candidate.estimatedCost.approximateTokens, 0),
    },
  };
};

const modelCapabilities = (maxContextTokens: number, maxOutputTokens = 128) => ({
  streaming: true,
  tools: false,
  inputModalities: ['text'] as const,
  maxContextTokens,
  maxOutputTokens,
});

describe('AI context budgeting', () => {
  it('compacts old turns and stays inside the model-aware input budget', async () => {
    const messages: AiConversationMessage[] = [];
    for (let index = 0; index < 18; ++index) {
      messages.push(message(`u${index}`, 'user', `Question ${index} ${'u'.repeat(220)}`));
      messages.push(message(`a${index}`, 'assistant', `Answer ${index} ${'a'.repeat(220)}`));
    }
    messages.push(message('latest', 'user', `Latest request ${'z'.repeat(160)}`));

    const plan = await prepareAiContextBudget({
      conversation: conversation(messages),
      modelCapabilities: modelCapabilities(800),
      maxRecentTurns: 4,
      now: () => new Date('2026-10-03T02:00:00.000Z'),
    });

    expect(plan.diagnostics.modelContextTokens).toBe(800);
    expect(plan.diagnostics.estimatedInputTokens).toBeLessThanOrEqual(plan.diagnostics.inputBudgetTokens);
    expect(plan.diagnostics.compactedMessages).toBeGreaterThan(0);
    expect(plan.diagnostics.retainedRecentMessages).toBeLessThan(messages.length);
    expect(plan.summary?.throughMessageId).toBeTruthy();
    expect(plan.messages.some((candidate) => candidate.id === 'arc-context:conversation-summary')).toBe(true);
    expect(plan.messages.some((candidate) => candidate.id === 'latest')).toBe(true);
  });

  it('retains pinned context and distinguishes it from current explicit context', async () => {
    const pinned: AiConversationContextReference = {
      id: 'pinned-entity',
      kind: 'entity',
      label: 'Player Spawn',
      stableId: 'entity-guid-player-spawn',
    };
    const explicit: AiConversationContextReference = {
      id: 'explicit-asset',
      kind: 'asset',
      label: 'Hero Material',
      stableId: 'asset-guid-hero-material',
    };
    const messages = [
      message('old', 'user', `${'old '.repeat(300)}`),
      message('answer', 'assistant', `${'answer '.repeat(300)}`),
      message('latest', 'user', 'Update this material', [explicit]),
    ];

    const plan = await prepareAiContextBudget({
      conversation: conversation(messages, { pinnedContext: [pinned] }),
      modelCapabilities: modelCapabilities(2_000, 256),
      maxRecentTurns: 1,
    });

    const pinnedMessage = plan.messages.find((candidate) => candidate.id === 'arc-context:pinned');
    const explicitMessage = plan.messages.find((candidate) => candidate.id === 'arc-context:explicit');
    expect(JSON.stringify(pinnedMessage)).toContain('entity-guid-player-spawn');
    expect(JSON.stringify(explicitMessage)).toContain('asset-guid-hero-material');
    expect(plan.diagnostics.decisions.some((decision) => decision.origin === 'pinned' && decision.included)).toBe(true);
    expect(plan.diagnostics.decisions.some((decision) => decision.origin === 'explicit' && decision.included)).toBe(
      true,
    );
  });

  it('force-refreshes stale editor context and re-resolves a stable reference before use', async () => {
    const reference: AiConversationContextReference = {
      id: 'selected-entity',
      kind: 'entity',
      stableId: 'entity-guid-1',
      metadata: {
        projectGuid: 'project-a',
        revision: { sceneRevision: 1, worldEpoch: 7 },
      },
    };
    const snapshots = [
      contextSnapshot({ sceneRevision: 2, worldEpoch: 7, eventSequence: 10 }),
      contextSnapshot(
        { sceneRevision: 3, worldEpoch: 7, eventSequence: 11 },
        { entities: [{ guid: 'entity-guid-1', name: 'Cube' }] },
      ),
    ];
    const calls: Array<{ forceRefresh?: boolean } | undefined> = [];
    const source: AiProjectContextSource = {
      async collect(options) {
        calls.push(options);
        return snapshots[Math.min(calls.length - 1, snapshots.length - 1)]!;
      },
    };

    const plan = await prepareAiContextBudget({
      conversation: conversation([message('latest', 'user', 'Inspect the pinned entity')], {
        pinnedContext: [reference],
      }),
      modelCapabilities: modelCapabilities(4_000, 512),
      projectGuid: 'project-a',
      projectContextSource: source,
    });

    expect(calls).toHaveLength(2);
    expect(calls[1]).toEqual({ forceRefresh: true });
    expect(plan.diagnostics.refreshState).toBe('forced');
    expect(plan.diagnostics.rejectedReferences).toEqual([]);
    expect(JSON.stringify(plan.messages.find((candidate) => candidate.id === 'arc-context:pinned'))).toContain(
      'entity-guid-1',
    );
    expect(plan.messages.some((candidate) => candidate.id === 'arc-context:auto:scene')).toBe(true);
  });

  it('rejects stale references that still cannot be resolved after a refresh', async () => {
    const reference: AiConversationContextReference = {
      id: 'deleted-entity',
      kind: 'entity',
      stableId: 'deleted-guid',
      metadata: {
        projectGuid: 'project-a',
        revision: { sceneRevision: 1, worldEpoch: 3 },
      },
    };
    const snapshot = contextSnapshot({ sceneRevision: 5, worldEpoch: 4 }, { entities: [] });
    let callCount = 0;
    const source: AiProjectContextSource = {
      async collect() {
        ++callCount;
        return snapshot;
      },
    };

    const plan = await prepareAiContextBudget({
      conversation: conversation([message('latest', 'user', 'Where did it go?')], { pinnedContext: [reference] }),
      modelCapabilities: modelCapabilities(4_000, 512),
      projectGuid: 'project-a',
      projectContextSource: source,
    });

    expect(callCount).toBe(2);
    expect(plan.diagnostics.rejectedReferences).toEqual([
      expect.objectContaining({ id: 'deleted-entity', origin: 'pinned' }),
    ]);
    expect(plan.messages.some((candidate) => candidate.id === 'arc-context:pinned')).toBe(false);
  });

  it('refreshes cached automatic context when its freshness age crosses policy', async () => {
    const cached = contextSnapshot({ sceneRevision: 9, worldEpoch: 2 }, { entities: [] }, 'project-a', 5_000);
    const fresh = contextSnapshot({ sceneRevision: 9, worldEpoch: 2 }, { entities: [] }, 'project-a', 0);
    const calls: Array<{ forceRefresh?: boolean } | undefined> = [];
    const source: AiProjectContextSource = {
      async collect(options) {
        calls.push(options);
        return calls.length === 1 ? cached : fresh;
      },
    };

    const plan = await prepareAiContextBudget({
      conversation: conversation([message('latest', 'user', 'What is selected?')]),
      modelCapabilities: modelCapabilities(4_000, 512),
      projectGuid: 'project-a',
      projectContextSource: source,
      maxAutomaticContextAgeMs: 1_000,
    });

    expect(calls).toEqual([undefined, { forceRefresh: true }]);
    expect(plan.diagnostics.refreshState).toBe('forced');
    expect(plan.diagnostics.decisions.some((decision) => decision.origin === 'automatic' && decision.included)).toBe(
      true,
    );
  });
});
