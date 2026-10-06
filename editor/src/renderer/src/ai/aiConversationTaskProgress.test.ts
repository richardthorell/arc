import { describe, expect, it } from 'vitest';

import { recordConversationTaskUpdate } from './aiConversationTaskProgress';

describe('AI conversation task progress', () => {
  it('keeps a failed step failed and cancels downstream planned work when the model later reports success', () => {
    const running = recordConversationTaskUpdate(
      undefined,
      {
        id: 'material-plan',
        planId: 'material-plan',
        title: 'Apply materials',
        state: 'in_progress',
        children: [
          { id: 'assign', title: 'Assign materials', state: 'in_progress' },
          { id: 'verify', title: 'Verify assignments', state: 'planned' },
        ],
      },
      '2026-10-05T22:00:00Z',
    );

    const failed = recordConversationTaskUpdate(
      running,
      {
        id: 'material-plan',
        planId: 'material-plan',
        title: 'Apply materials',
        state: 'failed',
        children: [
          { id: 'assign', title: 'Assign materials', state: 'failed', detail: 'Materials could not be loaded' },
          { id: 'verify', title: 'Verify assignments', state: 'planned' },
        ],
      },
      '2026-10-05T22:00:01Z',
    );

    const contradictoryModelUpdate = recordConversationTaskUpdate(
      failed,
      {
        id: 'material-plan',
        planId: 'material-plan',
        title: 'Apply materials',
        state: 'completed',
        children: [
          { id: 'assign', title: 'Assign materials', state: 'completed' },
          { id: 'verify', title: 'Verify assignments', state: 'completed' },
        ],
      },
      '2026-10-05T22:00:02Z',
    );

    expect(contradictoryModelUpdate[0]).toMatchObject({
      state: 'failed',
      children: [
        { id: 'assign', state: 'failed' },
        { id: 'verify', state: 'cancelled' },
      ],
    });
  });

  it('replaces earlier generic fallback tasks when a semantic plan arrives', () => {
    const fallback = recordConversationTaskUpdate(
      undefined,
      {
        id: 'agent-step-0',
        title: 'Run 3 editor operations',
        state: 'completed',
        toolCallIds: ['call-1', 'call-2', 'call-3'],
      },
      '2026-10-05T22:00:00Z',
    );

    const planned = recordConversationTaskUpdate(
      fallback,
      {
        id: 'playground-plan',
        planId: 'playground-plan',
        title: 'Build playground',
        state: 'in_progress',
        children: [{ id: 'arrange', title: 'Design a fun arrangement', state: 'in_progress' }],
      },
      '2026-10-05T22:00:01Z',
    );

    expect(planned).toHaveLength(1);
    expect(planned[0]).toMatchObject({
      id: 'playground-plan',
      planId: 'playground-plan',
      children: [{ id: 'arrange', title: 'Design a fun arrangement' }],
    });
  });

  it('does not persist edit.cancel as a user-facing fallback task', () => {
    const current = recordConversationTaskUpdate(
      undefined,
      { id: 'material-plan', planId: 'material-plan', title: 'Apply materials', state: 'failed' },
      '2026-10-05T22:00:00Z',
    );

    const afterCleanup = recordConversationTaskUpdate(
      current,
      { id: 'agent-step-4', title: 'edit.cancel', state: 'in_progress', toolCallIds: ['cancel-1'] },
      '2026-10-05T22:00:01Z',
    );

    expect(afterCleanup).toEqual(current);
  });
});
