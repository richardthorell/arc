// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { EditorReferenceProvider } from '../services/EditorReferenceContext';
import { UiAgentAssetChoiceCard } from './UiAgentChoiceCard';

afterEach(cleanup);

describe('UiAgentAssetChoiceCard', () => {
  it('resolves asset thumbnails, selects one option, and confirms the stable ARC URI', async () => {
    const onChoose = vi.fn();
    const controller = {
      resolve: vi.fn(async (reference: { kind: 'entity' | 'asset' | 'scene'; id: string }) => ({
        reference,
        label: reference.id === 'rock-a' ? 'Granite Rock 03' : 'Cliff Rock Large',
        subtitle: 'Model',
        thumbnailUrl: `data:image/png;base64,${reference.id}`,
      })),
      activate: vi.fn(),
      focus: vi.fn(),
      highlight: vi.fn(),
    };

    render(
      <EditorReferenceProvider controller={controller}>
        <UiAgentAssetChoiceCard
          title="Choose a rock"
          prompt="Pick one before placement."
          options={[
            { uri: 'arc://asset/rock-a', label: 'Granite Rock 03', reason: 'Closest silhouette.' },
            { uri: 'arc://asset/rock-b', label: 'Cliff Rock Large', reason: 'Larger foreground shape.' },
          ]}
          onChoose={onChoose}
        />
      </EditorReferenceProvider>,
    );

    expect(screen.getByText('Choose a rock')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Use this' })).toBeDisabled();

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Choose Granite Rock 03' }).querySelector('img')).toHaveAttribute(
        'src',
        'data:image/png;base64,rock-a',
      ),
    );

    fireEvent.click(screen.getByRole('button', { name: 'Choose Granite Rock 03' }));
    expect(screen.getByRole('button', { name: 'Choose Granite Rock 03' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Use this' })).toBeEnabled();

    fireEvent.click(screen.getByRole('button', { name: 'Use this' }));
    expect(onChoose).toHaveBeenCalledWith('arc://asset/rock-a');
  });

  it('keeps malformed non-asset references unavailable', () => {
    const controller = {
      resolve: vi.fn(),
      activate: vi.fn(),
    };

    render(
      <EditorReferenceProvider controller={controller}>
        <UiAgentAssetChoiceCard
          title="Choose"
          options={[
            { uri: 'arc://entity/entity-a', label: 'Entity' },
            { uri: 'not-a-uri', label: 'Broken' },
          ]}
          onChoose={vi.fn()}
        />
      </EditorReferenceProvider>,
    );

    expect(screen.getByRole('button', { name: 'Choose Entity' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Choose Broken' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Use this' })).toBeDisabled();
  });
});
