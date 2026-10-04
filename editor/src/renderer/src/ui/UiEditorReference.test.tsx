// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { EditorReferenceProvider } from '../services/EditorReferenceContext';
import { createEditorReferenceController } from '../services/editorReferences';
import { UiEditorReference } from './UiEditorReference';

afterEach(() => cleanup());

describe('UiEditorReference', () => {
  it('resolves a reference and routes click, double-click, and hover actions', async () => {
    const activateEntity = vi.fn();
    const focusEntity = vi.fn();
    const highlightEntity = vi.fn();
    const controller = createEditorReferenceController({
      resolveEntity: (id) => ({ label: 'Player', subtitle: `Entity · ${id}` }),
      activateEntity,
      focusEntity,
      highlightEntity,
    });

    render(
      <EditorReferenceProvider controller={controller}>
        <UiEditorReference href="arc://entity/player-guid" />
      </EditorReferenceProvider>,
    );

    const reference = await screen.findByRole('button', { name: 'entity reference: Player' });
    expect(reference).toHaveTextContent('Player');
    expect(reference).toHaveTextContent('Entity · player-guid');

    fireEvent.mouseEnter(reference);
    fireEvent.click(reference);
    fireEvent.doubleClick(reference);
    fireEvent.mouseLeave(reference);

    expect(activateEntity).toHaveBeenCalledWith('player-guid');
    expect(focusEntity).toHaveBeenCalledWith('player-guid');
    expect(highlightEntity).toHaveBeenNthCalledWith(1, 'player-guid', true);
    expect(highlightEntity).toHaveBeenNthCalledWith(2, 'player-guid', false);
  });

  it('uses author-provided link text while retaining resolved metadata', async () => {
    const controller = createEditorReferenceController({
      resolveAsset: () => ({ label: 'Brushed Metal', subtitle: 'Material' }),
    });

    render(
      <EditorReferenceProvider controller={controller}>
        <UiEditorReference href="arc://asset/material-guid">this material</UiEditorReference>
      </EditorReferenceProvider>,
    );

    const reference = await screen.findByRole('button', { name: 'asset reference: Brushed Metal' });
    expect(reference).toHaveTextContent('this material');
    expect(reference).toHaveTextContent('Material');
  });

  it('renders malformed ARC links as inert text', () => {
    render(<UiEditorReference href="arc://entity/">broken reference</UiEditorReference>);

    expect(screen.getByText('broken reference')).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });
});
