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

  it('composes a basic text response card on top of the base shell', () => {
    const { container } = render(<UiAgentTextCard state="streaming" text="Working on it" title="ARC" />);

    expect(screen.getByText('Working on it')).toBeInTheDocument();
    expect(container.querySelector('.ui-agent-card')).toHaveAttribute('data-state', 'streaming');
  });
});
