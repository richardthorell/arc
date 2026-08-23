// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiNodeCard } from './UiNodeCard';

afterEach(cleanup);

describe('UiNodeCard', () => {
  it('provides shared selected, accent, header, and badge chrome', () => {
    const onHeaderPointerDown = vi.fn();
    render(
      <UiNodeCard
        badge="P"
        badgeTitle="Parameter: Base Color"
        heading="Vector 3 / Color"
        onHeaderPointerDown={onHeaderPointerDown}
        selected
        tone="accent"
      >
        <div>Node body</div>
      </UiNodeCard>,
    );

    const card = screen.getByText('Node body').closest('article');
    expect(card).toHaveClass('ui-node-card', 'ui-node-card-accent', 'is-selected');
    expect(screen.getByText('P')).toHaveAttribute('title', 'Parameter: Base Color');

    fireEvent.pointerDown(screen.getByText('Vector 3 / Color').closest('header')!);
    expect(onHeaderPointerDown).toHaveBeenCalledTimes(1);
  });
});
