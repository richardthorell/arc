// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { MenuBar } from './MenuBar';

describe('MenuBar', () => {
  it('exposes the grid as a checked View menu command', () => {
    const toggle = vi.fn();
    render(<MenuBar projectTitle="Scene" onCommand={vi.fn()} gridVisible onToggleGrid={toggle} />);

    fireEvent.click(screen.getByRole('button', { name: 'View' }));
    const grid = screen.getByRole('menuitemcheckbox', { name: /Grid/ });
    expect(grid).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(grid);
    expect(toggle).toHaveBeenCalledOnce();
  });
});
