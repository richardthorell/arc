// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import {
  UiAgentApprovalCard,
  UiAgentAssetCard,
  UiAgentDiffCard,
  UiAgentErrorCard,
  UiAgentTaskCard,
  UiAgentToolCard,
  UiAgentViewportCard,
} from './UiAgentActivityCard';

afterEach(cleanup);

describe('UiAgentActivityCard', () => {
  it('shares status and disclosure behavior across structured response cards', () => {
    const { container } = render(
      <UiAgentToolCard
        details={<pre>{'{ "sceneRevision": 42 }'}</pre>}
        metadata="Step 2 · 84 ms"
        state="running"
        summary="Reading authoritative scene state."
        title="scene.overview"
      />,
    );

    const card = container.querySelector('[data-activity-kind="tool"]');
    expect(card).toHaveAttribute('data-activity-state', 'running');
    expect(screen.getByText('Running')).toBeVisible();
    expect(screen.getByText('Step 2 · 84 ms')).toBeVisible();
    expect(screen.queryByText('{ "sceneRevision": 42 }')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Expand details' }));
    expect(screen.getByText('{ "sceneRevision": 42 }')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Collapse details' })).toHaveAttribute('aria-expanded', 'true');
  });

  it('provides the complete structured response-card family', () => {
    const { container } = render(
      <>
        <UiAgentTaskCard title="Task" summary="Planning scene polish." state="pending" />
        <UiAgentToolCard title="Tool" summary="Reading scene." />
        <UiAgentApprovalCard title="Approval" summary="Waiting for approval." state="pending" />
        <UiAgentDiffCard title="Diff" summary="Three entities changed." />
        <UiAgentViewportCard title="Viewport" summary="Capture ready." />
        <UiAgentAssetCard title="Asset" summary="Using SM_Cabin." />
        <UiAgentErrorCard title="Error" summary="Operation failed." />
      </>,
    );

    for (const kind of ['task', 'tool', 'approval', 'diff', 'viewport', 'asset', 'error']) {
      expect(container.querySelector(`[data-activity-kind="${kind}"]`)).toBeInTheDocument();
    }
    expect(container.querySelector('[data-activity-kind="error"]')).toHaveAttribute('data-activity-state', 'error');
  });
});
