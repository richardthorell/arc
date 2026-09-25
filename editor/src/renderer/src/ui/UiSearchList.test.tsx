// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiSearchList } from './UiSearchList';

const items = [
  {
    id: 'asset:wood',
    variant: 'asset' as const,
    title: 'M_Wood',
    subtitle: 'Assets/Materials/M_Wood.arcmat',
    meta: 'material',
    state: 'ready',
  },
  {
    id: 'command:save',
    variant: 'command' as const,
    title: 'Save Scene',
    subtitle: 'Save the active scene.',
    meta: 'File',
    shortcut: 'Ctrl+S',
  },
];

afterEach(cleanup);

describe('UiSearchList', () => {
  it('renders asset and command result metadata', () => {
    render(<UiSearchList items={items} onActivate={() => undefined} />);

    expect(screen.getByText('M_Wood')).toBeInTheDocument();
    expect(screen.getByText('material')).toBeInTheDocument();
    expect(screen.getByText('ready')).toBeInTheDocument();
    expect(screen.getByText('Save Scene')).toBeInTheDocument();
    expect(screen.getByText('Ctrl+S')).toBeInTheDocument();
  });

  it('activates enabled results and leaves disabled commands inert', () => {
    const onActivate = vi.fn();
    const disabled = {
      ...items[1],
      id: 'command:undo',
      title: 'Undo',
      disabled: true,
      disabledReason: 'There is nothing to undo',
    };
    render(<UiSearchList items={[items[0], disabled]} onActivate={onActivate} />);

    fireEvent.click(screen.getByRole('button', { name: 'M_Wood' }));
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }));

    expect(onActivate).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Undo' })).toHaveAttribute('title', 'There is nothing to undo');
  });

  it('shows an explicit empty state', () => {
    render(<UiSearchList items={[]} emptyMessage="No commands found" onActivate={() => undefined} />);

    expect(screen.getByRole('status')).toHaveTextContent('No commands found');
  });
});
