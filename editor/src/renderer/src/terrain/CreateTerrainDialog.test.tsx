// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { CreateTerrainDialog } from './CreateTerrainDialog';

afterEach(cleanup);

describe('CreateTerrainDialog', () => {
  it('defaults to a flat 257 terrain and publishes through the terrain command', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    const created = vi.fn();
    render(<CreateTerrainDialog command={command} onClose={vi.fn()} onCreated={created} />);

    expect(screen.getByLabelText('Create terrain')).toHaveTextContent('CPU 0.5 MiB');
    expect(screen.getByLabelText('Source')).toHaveClass('ui-button');
    expect(screen.getByLabelText('Physical Size (m)')).toHaveValue('180');
    fireEvent.click(screen.getByRole('button', { name: 'Create Terrain' }));

    await waitFor(() =>
      expect(command).toHaveBeenCalledWith(
        'terrain.create',
        expect.objectContaining({
          source: 'flat',
          resolution: 257,
          patchQuads: 32,
          size: 180,
        }),
      ),
    );
    expect(created).toHaveBeenCalledOnce();
  });

  it('shows procedural seed controls and prevents operations over the undo budget', () => {
    render(<CreateTerrainDialog command={vi.fn()} onClose={vi.fn()} onCreated={vi.fn()} />);

    fireEvent.click(screen.getByLabelText('Source'));
    fireEvent.click(screen.getByRole('option', { name: 'Domain Warped' }));
    expect(screen.getByLabelText('Seed')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Resolution'));
    fireEvent.click(screen.getByRole('option', { name: '4097 x 4097' }));
    expect(screen.getByText(/exceeds the 64 MiB undo budget/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Terrain' })).toBeDisabled();
  });
});
