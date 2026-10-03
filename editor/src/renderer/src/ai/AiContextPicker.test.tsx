// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useState } from 'react';

import type { AiConversationContextReference } from '../../../common/aiConversationTypes';
import type { AiProjectContextSnapshot } from '../../../common/aiContextTypes';
import { AiContextChips, AiContextPicker } from './AiContextPicker';

const snapshot: AiProjectContextSnapshot = {
  schemaVersion: 1,
  collectionId: 'picker-collection',
  projectGuid: 'project-guid',
  capturedAt: '2026-10-03T00:00:00.000Z',
  revision: { sceneRevision: 2, eventSequence: 8 },
  sections: [
    {
      id: 'selection',
      status: 'ready',
      data: { guid: 'entity-guid', selectedGuids: ['entity-guid'] },
      truncated: false,
      freshness: { capturedAt: '2026-10-03T00:00:00.000Z', ageMs: 0, cache: 'live' },
      estimatedCost: { characters: 10, approximateTokens: 3 },
    },
    {
      id: 'scene',
      status: 'ready',
      data: { sceneGuid: 'scene-guid', entities: [{ guid: 'entity-guid', name: 'Hero Camera' }] },
      truncated: false,
      freshness: { capturedAt: '2026-10-03T00:00:00.000Z', ageMs: 0, cache: 'live' },
      estimatedCost: { characters: 10, approximateTokens: 3 },
    },
    {
      id: 'assets',
      status: 'ready',
      data: { assets: [{ guid: 'asset-guid', path: 'Content/HeroRock.glb', typeId: 'arc.mesh' }] },
      truncated: false,
      freshness: { capturedAt: '2026-10-03T00:00:00.000Z', ageMs: 0, cache: 'live' },
      estimatedCost: { characters: 10, approximateTokens: 3 },
    },
  ],
  estimatedCost: { characters: 30, approximateTokens: 9 },
};

const PickerHarness = () => {
  const [selected, setSelected] = useState<AiConversationContextReference[]>([]);
  return (
    <>
      <AiContextPicker
        source={{ collect: vi.fn().mockResolvedValue(snapshot) }}
        selected={selected}
        supportsImages={false}
        onAdd={(reference) => setSelected((current) => [...current, reference])}
        onClose={vi.fn()}
      />
      <AiContextChips
        references={selected}
        onRemove={(id) => setSelected((current) => current.filter((reference) => reference.id !== id))}
      />
    </>
  );
};

afterEach(cleanup);

describe('AiContextPicker', () => {
  it('adds visible removable context chips from structured project context', async () => {
    render(<PickerHarness />);

    await waitFor(() => expect(screen.getByRole('button', { name: /^Current selection\b/ })).toBeVisible());
    fireEvent.click(screen.getByRole('button', { name: /^Current selection\b/ }));

    expect(screen.getByLabelText('Attached context')).toHaveTextContent('Current selection');
    expect(screen.getByRole('button', { name: 'Remove Current selection' })).toBeVisible();
    expect(screen.getByRole('button', { name: /^Current selection\b/ })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Remove Current selection' }));
    expect(screen.queryByLabelText('Attached context')).not.toBeInTheDocument();
  });

  it('searches entity and asset candidates and explains image capability requirements', async () => {
    render(<PickerHarness />);
    await waitFor(() => expect(screen.getByLabelText('Search context')).toBeVisible());

    fireEvent.change(screen.getByLabelText('Search context'), { target: { value: 'rock' } });
    expect(screen.getByRole('button', { name: /HeroRock/ })).toBeVisible();
    expect(screen.queryByRole('button', { name: /Hero Camera/ })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Search context'), { target: { value: '' } });
    const capture = screen.getByRole('button', { name: /Viewport capture/ });
    expect(capture).toBeDisabled();
    expect(capture).toHaveAttribute('title', 'The selected model does not accept image input');
  });
});
