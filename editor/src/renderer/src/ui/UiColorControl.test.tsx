// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiColorControl } from './UiColorControl';

afterEach(cleanup);

describe('UiColorControl', () => {
  it('shows the channel type and opens the shared picker', () => {
    const onCommit = vi.fn();
    render(<UiColorControl label="Base Color" value={{ x: 0.4, y: 0.2, z: 0.1, w: 0.5 }} onCommit={onCommit} />);

    expect(screen.getByText('RGBA')).toBeTruthy();
    expect(screen.queryByText('HDR')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Open Base Color color picker' }));
    expect(screen.getByRole('dialog', { name: 'Base Color color picker' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'OK' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Color representation' })).toBeTruthy();
    expect(screen.getByRole('slider', { name: 'Color R' })).toBeTruthy();
    expect(screen.getByRole('slider', { name: 'Color G' })).toBeTruthy();
    expect(screen.getByRole('slider', { name: 'Color B' })).toBeTruthy();
    expect(screen.getByRole('slider', { name: 'Color A' })).toBeTruthy();
    expect(screen.queryByText('Linear storage')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Close dialog' })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onCommit).toHaveBeenCalledWith({ x: 0.4, y: 0.2, z: 0.1, w: 0.5 });
    expect(screen.queryByRole('dialog', { name: 'Base Color color picker' })).toBeNull();
  });

  it('shows the classic color presets with named options', () => {
    render(<UiColorControl label="Base Color" value={{ x: 0.4, y: 0.2, z: 0.1, w: 1 }} onCommit={() => undefined} />);

    fireEvent.click(screen.getByRole('button', { name: 'Open Base Color color picker' }));
    fireEvent.click(screen.getByRole('button', { name: 'Color preset' }));

    for (const name of [
      'Black',
      'Silver',
      'Gray',
      'White',
      'Maroon',
      'Red',
      'Purple',
      'Fuchsia',
      'Green',
      'Lime',
      'Olive',
      'Yellow',
      'Navy',
      'Blue',
      'Teal',
      'Aqua',
    ]) {
      expect(screen.getByRole('option', { name })).toBeTruthy();
    }
  });

  it('shows RGB and HDR for a high-dynamic-range color without alpha', () => {
    render(
      <UiColorControl
        allowAlpha={false}
        label="Emissive"
        maxChannelValue={16}
        value={{ x: 4, y: 1, z: 0.25, w: 1 }}
        onCommit={() => undefined}
      />,
    );

    expect(screen.getByText('RGB')).toBeTruthy();
    expect(screen.getByText('HDR')).toBeTruthy();
  });
});
