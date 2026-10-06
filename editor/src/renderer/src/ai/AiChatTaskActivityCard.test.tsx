// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { AiChatTaskActivityCard } from './AiChatActivityCards';

afterEach(cleanup);

describe('AiChatTaskActivityCard', () => {
  it('collapses completed task history behind a Show tasks action', async () => {
    const { container, rerender } = render(
      <AiChatTaskActivityCard
        reference={{
          id: 'agent-step-0',
          title: 'Applying editor changes',
          state: 'in_progress',
          step: 0,
          toolCallIds: ['create', 'rename'],
          startedAt: '2026-10-05T05:00:00Z',
        }}
      />,
    );

    expect(screen.getByRole('status', { name: 'AI progress' })).toBeVisible();
    expect(screen.getByText('Applying editor changes')).toBeVisible();
    expect(container.querySelector('[data-progress-state="working"]')).toBeInTheDocument();
    expect(container.querySelector('[data-activity-kind="task"]')).not.toBeInTheDocument();

    rerender(
      <AiChatTaskActivityCard
        reference={{
          id: 'agent-step-0',
          title: 'Applying editor changes',
          state: 'completed',
          step: 0,
          toolCallIds: ['create', 'rename'],
          startedAt: '2026-10-05T05:00:00Z',
          completedAt: '2026-10-05T05:00:02Z',
        }}
      />,
    );

    await waitFor(() => expect(screen.getByRole('button', { name: 'Show tasks' })).toBeVisible());
    expect(screen.queryByRole('status', { name: 'AI progress' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Show tasks' }));
    expect(screen.getByRole('button', { name: 'Hide tasks' })).toBeVisible();
    expect(screen.getByText('Applying editor changes')).toBeVisible();
    expect(container.querySelector('[data-progress-state="complete"] svg')).not.toBeInTheDocument();
  });

  it('keeps completed plan steps as rings while the larger spinner advances to the next row', () => {
    const { container, rerender } = render(
      <AiChatTaskActivityCard
        reference={{
          id: 'capsule-plan',
          planId: 'capsule-plan',
          title: 'Create green capsule',
          state: 'in_progress',
          children: [
            { id: 'inspect', title: 'Inspect scene', state: 'completed' },
            {
              id: 'build',
              title: 'Build capsule',
              state: 'in_progress',
              children: [
                { id: 'create', title: 'Create capsule', state: 'in_progress', toolCallIds: ['batch-1'] },
                { id: 'verify', title: 'Verify result', state: 'planned' },
              ],
            },
          ],
        }}
      />,
    );

    expect(screen.getByText('Inspect scene')).toBeVisible();
    expect(screen.getByText('Create capsule')).toBeVisible();
    expect(screen.getByText('Verify result')).toBeVisible();
    expect(container.querySelectorAll('[data-progress-state="complete"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-progress-state="working"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-progress-state="waiting"]')).toHaveLength(1);
    expect(container.querySelector('[data-progress-state="complete"] svg')).not.toBeInTheDocument();

    rerender(
      <AiChatTaskActivityCard
        reference={{
          id: 'capsule-plan',
          planId: 'capsule-plan',
          title: 'Create green capsule',
          state: 'in_progress',
          children: [
            { id: 'inspect', title: 'Inspect scene', state: 'completed' },
            {
              id: 'build',
              title: 'Build capsule',
              state: 'in_progress',
              children: [
                { id: 'create', title: 'Create capsule', state: 'completed', toolCallIds: ['batch-1'] },
                { id: 'verify', title: 'Verify result', state: 'in_progress' },
              ],
            },
          ],
        }}
      />,
    );

    expect(screen.getByText('Inspect scene')).toBeVisible();
    expect(screen.getByText('Create capsule')).toBeVisible();
    expect(screen.getByText('Verify result')).toBeVisible();
    expect(container.querySelectorAll('[data-progress-state="complete"]')).toHaveLength(2);
    expect(container.querySelectorAll('[data-progress-state="working"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-progress-state="waiting"]')).toHaveLength(0);
  });

  it('opens failed task history by default with a red X and no fake active row', () => {
    const { container } = render(
      <AiChatTaskActivityCard
        reference={{
          id: 'capsule-plan',
          planId: 'capsule-plan',
          title: 'Create green capsule',
          state: 'failed',
          children: [
            { id: 'inspect', title: 'Inspect scene', state: 'completed' },
            { id: 'create', title: 'Create capsule', state: 'failed', detail: 'Editor operation failed' },
            { id: 'verify', title: 'Verify result', state: 'cancelled' },
          ],
        }}
      />,
    );

    expect(screen.getByRole('button', { name: 'Hide tasks' })).toBeVisible();
    expect(screen.getByText('Inspect scene')).toBeVisible();
    expect(screen.getByText('Create capsule')).toBeVisible();
    expect(screen.queryByText('Verify result')).not.toBeInTheDocument();
    expect(container.querySelectorAll('[data-progress-state="complete"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-progress-state="failed"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-progress-state="working"]')).toHaveLength(0);
    expect(container.querySelectorAll('[data-progress-state="waiting"]')).toHaveLength(0);
    expect(container.querySelector('[data-progress-state="failed"] svg')).toBeInTheDocument();
  });
});
