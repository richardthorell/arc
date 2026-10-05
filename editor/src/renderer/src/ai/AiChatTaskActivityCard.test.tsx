// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { AiChatTaskActivityCard } from './AiChatActivityCards';

afterEach(cleanup);

describe('AiChatTaskActivityCard', () => {
  it('shows a compact live row while a task is active and removes it when complete', () => {
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

    expect(screen.queryByRole('status', { name: 'AI progress' })).not.toBeInTheDocument();
  });

  it('shows the current semantic plan step with queued work waiting behind it', () => {
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

    expect(screen.queryByText('Inspect scene')).not.toBeInTheDocument();
    expect(screen.getByText('Create capsule')).toBeVisible();
    expect(screen.getByText('Verify result')).toBeVisible();
    expect(screen.getByText('Waiting')).toBeVisible();
    expect(container.querySelectorAll('[data-progress-state="working"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-progress-state="waiting"]')).toHaveLength(1);

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

    expect(screen.queryByText('Create capsule')).not.toBeInTheDocument();
    expect(screen.getByText('Verify result')).toBeVisible();
    expect(container.querySelectorAll('[data-progress-state="working"]')).toHaveLength(1);
    expect(container.querySelectorAll('[data-progress-state="waiting"]')).toHaveLength(0);
  });
});
