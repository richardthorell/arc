// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

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

  it('supports content-only speech cards with speaker side and tone', () => {
    const { container } = render(
      <UiAgentTextCard side="left" state="streaming" text="Working on it" tone="agent" />,
    );

    expect(screen.getByText('Working on it')).toBeInTheDocument();
    const card = container.querySelector('.ui-agent-card');
    expect(card).toHaveAttribute('data-state', 'streaming');
    expect(card).toHaveAttribute('data-side', 'left');
    expect(card).toHaveAttribute('data-tone', 'agent');
    expect(container.querySelector('.ui-agent-card-header')).not.toBeInTheDocument();
  });

  it('can place the speech tab on the user side', () => {
    const { container } = render(<UiAgentTextCard side="right" text="My prompt" tone="user" />);

    const card = container.querySelector('.ui-agent-card');
    expect(card).toHaveAttribute('data-side', 'right');
    expect(card).toHaveAttribute('data-tone', 'user');
  });
});
