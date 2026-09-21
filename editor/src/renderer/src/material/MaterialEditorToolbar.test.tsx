// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { MaterialEditorToolbar } from './MaterialEditorToolbar';
import { createDefaultMaterialGraph } from './materialGraphTypes';

const materialState = vi.hoisted(() => ({
  compileMaterialDocument: vi.fn(async () => true),
  redoMaterialGraph: vi.fn(),
  reloadMaterialDocument: vi.fn(async () => true),
  saveAndPublishMaterialDocument: vi.fn(async () => true),
  saveMaterialDocument: vi.fn(async () => true),
  setMaterialGraphView: vi.fn(),
  setMaterialLiveUpdate: vi.fn(),
  undoMaterialGraph: vi.fn(),
  useMaterialDocumentState: vi.fn(),
}));

vi.mock('./materialDocumentState', () => materialState);

const document: EditorDocument = {
  id: 'material-toolbar-test',
  kind: 'material',
  title: 'Test Material',
  path: 'Content/Test.arcmat',
  assetGuid: '00112233445566778899aabbccddeeff',
  assetScope: 'project',
  dirty: true,
  readOnly: false,
};

const graph = createDefaultMaterialGraph();

beforeEach(() => {
  for (const value of Object.values(materialState)) if ('mockClear' in value) value.mockClear();
  materialState.useMaterialDocumentState.mockReturnValue({
    asset: { version: 4, graph },
    graph,
    loading: false,
    saving: false,
    compiling: false,
    liveUpdate: true,
    showGrid: true,
    dimUnrelated: false,
    history: [graph],
    historyIndex: 0,
    compilation: {
      status: 'succeeded',
      succeeded: true,
      diagnostics: [],
    },
  });
});

afterEach(cleanup);

describe('MaterialEditorToolbar', () => {
  it('uses the compact material-specific toolbar without document identity labels', () => {
    render(<MaterialEditorToolbar document={{ ...document, readOnly: true }} />);

    expect(screen.getByRole('button', { name: /Save/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Compiled' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Live Update/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Stats/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /View/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /More/ })).toBeInTheDocument();

    expect(screen.queryByText('Read-only')).not.toBeInTheDocument();
    expect(screen.queryByText('Native ready')).not.toBeInTheDocument();
    expect(screen.queryByText('Material', { selector: '.material-document-toolbar-label' })).not.toBeInTheDocument();
  });

  it('controls live update and graph view options', () => {
    render(<MaterialEditorToolbar document={document} />);

    fireEvent.click(screen.getByRole('button', { name: /Live Update/ }));
    const liveMenu = screen.getByRole('menu', { name: 'Live update options' });
    fireEvent.click(within(liveMenu).getByRole('menuitem', { name: 'Paused' }));
    expect(materialState.setMaterialLiveUpdate).toHaveBeenCalledWith(document, false);

    fireEvent.click(screen.getByRole('button', { name: /View/ }));
    const viewMenu = screen.getByRole('menu', { name: 'Material graph view options' });
    fireEvent.click(within(viewMenu).getByRole('menuitem', { name: 'Show Grid' }));
    expect(materialState.setMaterialGraphView).toHaveBeenCalledWith(document, { showGrid: false });
  });

  it('shows concise graph/compiler stats', () => {
    render(<MaterialEditorToolbar document={document} />);

    fireEvent.click(screen.getByRole('button', { name: /Stats/ }));
    const stats = screen.getByRole('dialog', { name: 'Material stats' });

    expect(within(stats).getByText('Nodes')).toBeInTheDocument();
    expect(within(stats).getByText('Connections')).toBeInTheDocument();
    expect(within(stats).getByText('Diagnostics')).toBeInTheDocument();
    expect(within(stats).getByText('0 errors · 0 warnings')).toBeInTheDocument();
  });
});
