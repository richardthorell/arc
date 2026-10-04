// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import type { AiConversationToolReference } from '../../../common/aiConversationTypes';
import { AiChatToolActivityCard } from './AiChatActivityCards';

afterEach(cleanup);

const reference = (overrides: Partial<AiConversationToolReference> = {}): AiConversationToolReference => ({
  toolCallId: 'tool-1',
  name: 'scene.overview',
  state: 'complete',
  step: 1,
  startedAt: '2026-10-04T08:00:00.000Z',
  completedAt: '2026-10-04T08:00:00.840Z',
  resultContent: '{"sceneRevision":42}',
  ...overrides,
});

describe('AiChatToolActivityCard', () => {
  it('renders stable operation metadata and bounded details from persisted tool references', () => {
    const { container } = render(<AiChatToolActivityCard reference={reference()} />);

    expect(container.querySelector('[data-activity-kind="tool"]')).toBeInTheDocument();
    expect(screen.getByText('scene.overview')).toBeVisible();
    expect(screen.getByText('Step 2 · 840 ms')).toBeVisible();
    expect(screen.queryByText(/sceneRevision/u)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Expand details' }));
    expect(screen.getByText(/sceneRevision/u)).toBeVisible();
  });

  it('specializes asset, viewport, and error tool activity', () => {
    const { container, rerender } = render(
      <AiChatToolActivityCard reference={reference({ name: 'assets.list', operation: 'assets.list' })} />,
    );
    expect(container.querySelector('[data-activity-kind="asset"]')).toBeInTheDocument();

    rerender(<AiChatToolActivityCard reference={reference({ name: 'viewport.observe' })} />);
    expect(container.querySelector('[data-activity-kind="viewport"]')).toBeInTheDocument();

    rerender(
      <AiChatToolActivityCard
        reference={reference({
          state: 'error',
          errorCode: 'tool_error',
          retryable: true,
          resultContent: 'Viewport capture failed',
        })}
      />,
    );
    expect(container.querySelector('[data-activity-state="error"]')).toBeInTheDocument();
    expect(screen.getByText(/Viewport capture failed/u)).toBeVisible();
  });
});
