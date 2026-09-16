// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiNodeCard } from './UiNodeCard';

afterEach(cleanup);

describe('UiNodeCard', () => {
  it('provides shared selected, accent, header, badge, color, and icon chrome', () => {
    const onHeaderPointerDown = vi.fn();
    render(
      <UiNodeCard
        badge="P"
        badgeTitle="Parameter: Base Color"
        heading="Vector 3 / Color"
        icon={<span data-testid="node-card-icon">V</span>}
        nodeColor="#4d9ee8"
        onHeaderPointerDown={onHeaderPointerDown}
        selected
        tone="accent"
      >
        <div>Node body</div>
      </UiNodeCard>,
    );

    const card = screen.getByText('Node body').closest('article');
    expect(card).toHaveClass(
      'ui-node-card',
      'ui-node-card-accent',
      'ui-node-card-has-color',
      'ui-node-card-has-icon',
      'is-selected',
    );
    expect((card as HTMLElement).style.getPropertyValue('--ui-node-card-color')).toBe('#4d9ee8');
    expect(screen.getByTestId('node-card-icon').closest('.ui-node-card-icon')).toBeInTheDocument();
    expect(screen.getByText('P')).toHaveAttribute('title', 'Parameter: Base Color');

    fireEvent.pointerDown(screen.getByText('Vector 3 / Color').closest('header')!);
    expect(onHeaderPointerDown).toHaveBeenCalledTimes(1);
  });
});
