// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiVector3Control } from './UiVector3Control';

afterEach(cleanup);

const baseProps = {
  label: 'Location',
  precision: 2,
  scrubSensitivity: 0.05,
  step: 0.1,
  value: { x: 0, y: 2, z: 6 },
  onCommit: vi.fn(),
  onPreview: vi.fn(),
};

describe('UiVector3Control', () => {
  it('always reserves link and reset action slots so axis widths stay aligned', () => {
    const { container, rerender } = render(<UiVector3Control {...baseProps} showLabel={false} />);

    expect(container.querySelectorAll('.ui-vector3-action-slot')).toHaveLength(2);
    expect(screen.queryByRole('button', { name: /Link location axes/ })).toBeNull();

    rerender(
      <UiVector3Control
        {...baseProps}
        label="Scale"
        linkable
        linked
        showLabel={false}
        onReset={() => undefined}
        onToggleLinked={() => undefined}
      />,
    );

    expect(container.querySelectorAll('.ui-vector3-action-slot')).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Unlink scale axes' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Reset Scale' })).toBeTruthy();
  });

  it('commits an edited axis value through the shared numeric input', () => {
    const onCommit = vi.fn();
    render(<UiVector3Control {...baseProps} onCommit={onCommit} />);

    const x = screen.getByLabelText('Location X');
    fireEvent.change(x, { target: { value: '1.25' } });
    fireEvent.blur(x);

    expect(onCommit).toHaveBeenCalledWith('x', 1.25);
  });
});
