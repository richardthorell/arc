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
    fireEvent.change(screen.getByLabelText('Source'), { target: { value: 'procedural' } });
    expect(screen.getByLabelText('Seed')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Resolution'), { target: { value: '4097' } });
    expect(screen.getByText(/exceeds the 64 MiB undo budget/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Terrain' })).toBeDisabled();
  });
});
