// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { UiToggleButton } from './UiToggleButton';

describe('UiToggleButton', () => {
  it('exposes switch semantics and requests the next checked state', () => {
    const onCheckedChange = vi.fn();
    const { rerender } = render(
      <UiToggleButton aria-label="Two sided" checked={false} onCheckedChange={onCheckedChange} />,
    );

    const toggle = screen.getByRole('switch', { name: 'Two sided' });
    expect(toggle.getAttribute('aria-checked')).toBe('false');

    fireEvent.click(toggle);
    expect(onCheckedChange).toHaveBeenCalledWith(true);

    rerender(<UiToggleButton aria-label="Two sided" checked onCheckedChange={onCheckedChange} />);
    expect(toggle.getAttribute('aria-checked')).toBe('true');
  });

  it('can expose a visible accessible label without external wrapper markup', () => {
    const onCheckedChange = vi.fn();
    render(<UiToggleButton checked={false} label="Realtime" onCheckedChange={onCheckedChange} />);

    const toggle = screen.getByRole('switch', { name: 'Realtime' });
    expect(toggle.classList.contains('has-label')).toBe(true);
    fireEvent.click(toggle);
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });

  it('resolves a mixed state to enabled on click', () => {
    const onCheckedChange = vi.fn();
    render(<UiToggleButton aria-label="Mixed toggle" checked={false} mixed onCheckedChange={onCheckedChange} />);

    const toggle = screen.getByRole('switch', { name: 'Mixed toggle' });
    expect(toggle.classList.contains('is-mixed')).toBe(true);
    expect(toggle.getAttribute('data-mixed')).toBe('true');

    fireEvent.click(toggle);
    expect(onCheckedChange).toHaveBeenCalledWith(true);
  });

  it('does not change state while disabled', () => {
    const onCheckedChange = vi.fn();
    render(<UiToggleButton aria-label="Disabled toggle" checked={false} disabled onCheckedChange={onCheckedChange} />);

    fireEvent.click(screen.getByRole('switch', { name: 'Disabled toggle' }));
    expect(onCheckedChange).not.toHaveBeenCalled();
  });
});
