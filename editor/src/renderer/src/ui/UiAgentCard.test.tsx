// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiAgentCard, UiAgentTextCard } from './UiAgentCard';

afterEach(cleanup);

describe('UiAgentCard', () => {
  it('provides a reusable shell for richer agent responses', () => {
    render(
      <UiAgentCard actions={<button type="button">Inspect</button>} subtitle="Mock provider" title="ARC">
        <div>Custom response content</div>
      </UiAgentCard>,
    );

    expect(screen.getByText('ARC')).toBeInTheDocument();
    expect(screen.getByText('Mock provider')).toBeInTheDocument();
    expect(screen.getByText('Custom response content')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Inspect' })).toBeInTheDocument();
  });

  it('supports content-only speech cards with persistent left-aligned agent actions', () => {
    const { container } = render(
      <UiAgentTextCard side="left" state="streaming" text="Working on it" timestamp="2:15 PM" tone="agent" />,
    );

    expect(screen.getByText('Working on it')).toBeInTheDocument();
    expect(screen.getByText('2:15 PM')).toBeInTheDocument();
    const card = container.querySelector('.ui-agent-card');
    expect(card).toHaveAttribute('data-state', 'streaming');
    expect(card).toHaveAttribute('data-side', 'left');
    expect(card).toHaveAttribute('data-tone', 'agent');
    expect(card).toHaveAttribute('data-has-footer-actions', 'true');
    expect(container.querySelector('.ui-agent-card-header')).not.toBeInTheDocument();
    expect(container.querySelector('.ui-agent-card-timestamp')).toBeInTheDocument();
    expect(container.querySelector('.ui-agent-card-action-row')).toHaveAttribute('data-align', 'left');
    expect(container.querySelector('.ui-agent-card-action-row')).toHaveAttribute('data-reveal-on-hover', 'false');
    expect(screen.getByRole('button', { name: 'Copy message' })).toBeVisible();
  });

  it('right-aligns user actions and marks them for hover reveal', () => {
    const { container } = render(<UiAgentTextCard side="right" text="My prompt" tone="user" />);

    const card = container.querySelector('.ui-agent-card');
    expect(card).toHaveAttribute('data-side', 'right');
    expect(card).toHaveAttribute('data-tone', 'user');
    expect(container.querySelector('.ui-agent-card-action-row')).toHaveAttribute('data-align', 'right');
    expect(container.querySelector('.ui-agent-card-action-row')).toHaveAttribute('data-reveal-on-hover', 'true');
  });

  it('copies the message text represented by the card', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });
    render(<UiAgentTextCard side="left" text="**Rendered** message" tone="agent" />);

    fireEvent.click(screen.getByRole('button', { name: 'Copy message' }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith('**Rendered** message'));
    expect(screen.getByRole('button', { name: 'Copied' })).toBeVisible();
  });
});
