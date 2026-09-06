// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SettingsDialog } from './SettingsDialog';

beforeEach(() => {
  vi.stubGlobal('arc', {
    settings: {
      snapshot: vi.fn().mockResolvedValue({
        revision: 1,
        schema: [
          {
            key: 'renderer.defaultGrid',
            section: 'Renderer',
            label: 'Default Grid',
            description: 'Show the grid in new viewports.',
            type: 'boolean',
            defaultValue: true,
            scopes: ['user'],
          },
          {
            key: 'renderer.qualityTier',
            section: 'Renderer',
            label: 'Quality Tier',
            description: 'Renderer quality profile used by editor viewports.',
            type: 'enum',
            defaultValue: 'auto',
            options: ['auto', 'low', 'high'],
            scopes: ['user', 'project'],
          },
          {
            key: 'renderer.temporalHistoryWeight',
            section: 'Renderer',
            label: 'Temporal History Weight',
            description: 'Contribution retained from validated temporal history samples.',
            type: 'number',
            defaultValue: 0.9,
            minimum: 0,
            maximum: 1,
            step: 0.01,
            scopes: ['user', 'project'],
          },
        ],
        values: {
          'renderer.defaultGrid': true,
          'renderer.qualityTier': 'auto',
          'renderer.temporalHistoryWeight': 0.9,
        },
        sources: {
          'renderer.defaultGrid': 'default',
          'renderer.qualityTier': 'default',
          'renderer.temporalHistoryWeight': 'default',
        },
        restartRequired: [],
      }),
      update: vi.fn(),
    },
    recovery: {
      snapshot: vi.fn().mockResolvedValue({ uncleanShutdown: false, generations: [] }),
      restore: vi.fn(),
      discard: vi.fn(),
    },
    extensions: {
      snapshot: vi.fn().mockResolvedValue({ extensions: [] }),
    },
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('SettingsDialog', () => {
  it('renders a searchable hierarchical editor settings shell', async () => {
    const onResetLayout = vi.fn();
    render(<SettingsDialog onClose={vi.fn()} onResetLayout={onResetLayout} />);

    expect(screen.getByRole('dialog', { name: 'Editor Settings' })).toHaveAttribute('aria-modal', 'true');
    expect(screen.getByRole('button', { name: 'Close settings' })).toHaveFocus();
    expect(screen.getByRole('tree', { name: 'Settings sections' })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Editing/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Viewport/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /AI/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Source Control/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Platforms/ })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Settings scope' })).toHaveTextContent('User settings');
    expect(screen.getByRole('searchbox', { name: 'Search settings' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Reset workbench layout' }));
    expect(onResetLayout).toHaveBeenCalledTimes(1);

    await waitFor(() => expect(window.arc.settings.snapshot).toHaveBeenCalledTimes(1));
  });

  it('maps existing settings into the new hierarchy and searches descriptor text', async () => {
    render(<SettingsDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    await waitFor(() => expect(window.arc.settings.snapshot).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByRole('searchbox', { name: 'Search settings' }), { target: { value: 'default grid' } });
    expect(screen.getByRole('treeitem', { name: /Editing/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('treeitem', { name: /Viewport/ }));

    expect(screen.getByRole('heading', { name: 'Viewport' })).toBeInTheDocument();
    expect(screen.getByText('Default Grid')).toBeInTheDocument();
    expect(screen.queryByRole('treeitem', { name: /Platforms/ })).not.toBeInTheDocument();
  });

  it('uses shared controls without stealing focus when callback props change', async () => {
    const { rerender } = render(<SettingsDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    await waitFor(() => expect(window.arc.settings.snapshot).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole('treeitem', { name: /Viewport/ }));

    const qualityTier = screen.getByRole('combobox', { name: 'Quality Tier' });
    fireEvent.click(qualityTier);
    expect(screen.getByRole('listbox')).toBeInTheDocument();

    const historyWeight = screen.getByRole('spinbutton', { name: 'Temporal History Weight' });
    historyWeight.focus();
    expect(historyWeight).toHaveFocus();

    rerender(<SettingsDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    expect(historyWeight).toHaveFocus();
  });

  it('shows framework pages that do not have registered settings yet', () => {
    render(<SettingsDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);

    fireEvent.click(screen.getByRole('treeitem', { name: /Platforms/ }));
    expect(screen.getByRole('heading', { name: 'Platforms' })).toBeInTheDocument();
    expect(screen.getByText('No settings registered yet')).toBeInTheDocument();
  });

  it('closes from the close button, Escape, and backdrop', () => {
    const onClose = vi.fn();
    const { rerender } = render(<SettingsDialog onClose={onClose} onResetLayout={vi.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Close settings' }));
    expect(onClose).toHaveBeenCalledTimes(1);

    rerender(<SettingsDialog onClose={onClose} onResetLayout={vi.fn()} />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(2);

    const backdrop = screen.getByRole('dialog', { name: 'Editor Settings' }).parentElement;
    expect(backdrop).not.toBeNull();
    fireEvent.pointerDown(backdrop!);
    expect(onClose).toHaveBeenCalledTimes(3);
  });
});
