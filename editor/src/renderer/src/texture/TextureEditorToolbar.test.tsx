// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { TextureEditorToolbar } from './TextureEditorToolbar';
import { getTextureEditorViewState, setTextureEditorViewState } from './textureEditorViewState';

const document: EditorDocument = {
  id: 'texture:texture-guid',
  kind: 'texture',
  title: 'T_Rock.png',
  path: 'Content/Textures/T_Rock.png',
  assetGuid: 'texture-guid',
  assetScope: 'project',
  assetSnapshot: {
    id: 'texture-guid',
    guid: 'texture-guid',
    name: 'T_Rock.png',
    path: 'Content/Textures/T_Rock.png',
    scope: 'project',
    kind: 'texture',
    status: 'ready',
    readOnly: false,
    width: 2048,
    height: 1024,
    mipLevels: 12,
  },
  dirty: true,
  readOnly: false,
};

const documentState = vi.hoisted(() => ({
  hasPendingTextureSettings: vi.fn(() => true),
  revertTextureDocument: vi.fn(async () => true),
  saveTextureDocument: vi.fn(async () => true),
  useTextureDocumentState: vi.fn(() => ({
    pendingPatch: { brightness: 0.2 },
    saving: false,
    error: null,
  })),
}));

vi.mock('./textureDocumentState', () => documentState);

beforeEach(() => {
  for (const value of Object.values(documentState)) if ('mockClear' in value) value.mockClear();
  documentState.hasPendingTextureSettings.mockReturnValue(true);
  documentState.useTextureDocumentState.mockReturnValue({
    pendingPatch: { brightness: 0.2 },
    saving: false,
    error: null,
  });
  setTextureEditorViewState(document.id, {
    mipLevel: 0,
    previewMode: 'processed',
    exposure: 0,
    sampling: 'linear',
    channels: { r: true, g: true, b: true, a: true },
  });
});

afterEach(cleanup);

describe('TextureEditorToolbar', () => {
  it('uses left, center, and right toolbar groups with Save as the persistence action', () => {
    const { container } = render(<TextureEditorToolbar document={document} />);

    expect(container.querySelector('.toolbar-left')).toContainElement(
      screen.getByRole('button', { name: 'Save texture' }),
    );
    expect(container.querySelector('.toolbar-center')).toContainElement(
      screen.getByRole('button', { name: 'Processed' }),
    );
    expect(container.querySelector('.toolbar-right')).toContainElement(screen.getByRole('button', { name: /View/ }));

    expect(screen.getByRole('button', { name: 'Source' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Difference' })).toBeInTheDocument();
    expect(screen.getByLabelText('Texture channels')).toBeInTheDocument();
    expect(screen.getByText('Mip Level:')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Mip level' })).toHaveTextContent('0');
    expect(screen.queryByRole('button', { name: 'Decrease mip level' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Increase mip level' })).not.toBeInTheDocument();
    expect(screen.queryByText('Apply settings')).not.toBeInTheDocument();
    expect(screen.queryByText('Discard')).not.toBeInTheDocument();
  });

  it('shows mip dimensions as option subtitles without repeating the field label', () => {
    render(<TextureEditorToolbar document={document} />);

    fireEvent.click(screen.getByRole('combobox', { name: 'Mip level' }));
    const options = screen.getAllByRole('option');

    expect(options[0]).toHaveTextContent('0');
    expect(options[0]).toHaveTextContent('2048 × 1024');
    expect(options[1]).toHaveTextContent('1');
    expect(options[1]).toHaveTextContent('1024 × 512');
    expect(options[0]).not.toHaveTextContent('Mip Level');
  });

  it('saves directly and keeps revert under the Save dropdown', () => {
    render(<TextureEditorToolbar document={document} />);

    fireEvent.click(screen.getByRole('button', { name: 'Save texture' }));
    expect(documentState.saveTextureDocument).toHaveBeenCalledWith(document);

    fireEvent.click(screen.getByRole('button', { name: 'Texture save actions' }));
    const menu = screen.getByRole('menu');
    fireEvent.click(within(menu).getByRole('menuitem', { name: 'Revert Unsaved Changes' }));
    expect(documentState.revertTextureDocument).toHaveBeenCalledWith(document);
  });

  it('keeps preview-only sampling and exposure under View', () => {
    render(<TextureEditorToolbar document={document} />);

    fireEvent.click(screen.getByRole('button', { name: /View/ }));
    const view = screen.getByRole('dialog', { name: 'Texture preview view options' });

    expect(within(view).getByRole('combobox', { name: 'Texture preview sampling' })).toHaveTextContent('Linear');
    expect(within(view).getByLabelText('Texture preview exposure')).toHaveValue('0');

    fireEvent.click(screen.getByRole('button', { name: 'Source' }));
    expect(getTextureEditorViewState(document.id).previewMode).toBe('source');
  });
});
