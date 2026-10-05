// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { AiChatTaskActivityCard } from './AiChatActivityCards';

afterEach(cleanup);

describe('AiChatTaskActivityCard', () => {
  it('renders live progress and linked tool calls as one compact task card', () => {
    const { container, rerender } = render(
      <AiChatTaskActivityCard
        reference={{
          id: 'agent-step-0',
          title: 'Run 2 editor operations',
          state: 'in_progress',
          step: 0,
          toolCallIds: ['create', 'rename'],
          startedAt: '2026-10-05T05:00:00Z',
        }}
      />,
    );

    expect(container.querySelector('[data-activity-kind="task"]')).toHaveAttribute('data-activity-state', 'running');
    expect(screen.getByText('Run 2 editor operations')).toBeVisible();
    expect(screen.getByText('Step 1 · 2 tool calls')).toBeVisible();

    rerender(
      <AiChatTaskActivityCard
        reference={{
          id: 'agent-step-0',
          title: 'Run 2 editor operations',
          state: 'failed',
          step: 0,
          toolCallIds: ['create', 'rename'],
          detail: 'Failed while running edit.apply',
          startedAt: '2026-10-05T05:00:00Z',
          completedAt: '2026-10-05T05:00:02Z',
        }}
      />,
    );

    expect(container.querySelector('[data-activity-kind="task"]')).toHaveAttribute('data-activity-state', 'error');
    expect(screen.getByText('Failed while running edit.apply')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Collapse details' }));
    fireEvent.click(screen.getByRole('button', { name: 'Expand details' }));
    expect(screen.getByText('create, rename')).toBeVisible();
  });

  it('renders a semantic plan as one grouped card with nested steps', () => {
    const { container } = render(
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
                {
                  id: 'create',
                  title: 'Create capsule',
                  state: 'in_progress',
                  toolCallIds: ['batch-1'],
                },
                { id: 'verify', title: 'Verify result', state: 'planned' },
              ],
            },
          ],
        }}
      />,
    );

    expect(container.querySelectorAll('[data-activity-kind="task"]')).toHaveLength(1);
    expect(screen.getByText('Create green capsule')).toBeVisible();
    expect(screen.getByText('Working on Create capsule')).toBeVisible();
    expect(screen.getByText('1/3 steps complete')).toBeVisible();
    expect(screen.getByText('Inspect scene')).toBeVisible();
    expect(screen.getByText('Build capsule')).toBeVisible();
    expect(screen.getByText('Create capsule')).toBeVisible();
    expect(screen.getByText('Verify result')).toBeVisible();
    expect(screen.getByText('1 linked tool call')).toBeVisible();
  });
});
