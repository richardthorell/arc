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
