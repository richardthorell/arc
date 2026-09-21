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
    showStats: false,
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
    expect(screen.getByRole('button', { name: 'Material compile actions' })).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: 'Live Update' })).toBeChecked();
    expect(screen.getByRole('button', { name: /View/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Stats/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /More/ })).not.toBeInTheDocument();

    expect(screen.queryByText('Read-only')).not.toBeInTheDocument();
    expect(screen.queryByText('Native ready')).not.toBeInTheDocument();
    expect(screen.queryByText('Material', { selector: '.material-document-toolbar-label' })).not.toBeInTheDocument();
  });

  it('uses a direct live update toggle and keeps stats under View', () => {
    render(<MaterialEditorToolbar document={document} />);

    fireEvent.click(screen.getByRole('switch', { name: 'Live Update' }));
    expect(materialState.setMaterialLiveUpdate).toHaveBeenCalledWith(document, false);

    fireEvent.click(screen.getByRole('button', { name: /View/ }));
    const viewMenu = screen.getByRole('menu', { name: 'Material graph view options' });
    fireEvent.click(within(viewMenu).getByRole('menuitem', { name: 'Show Grid' }));
    expect(materialState.setMaterialGraphView).toHaveBeenCalledWith(document, { showGrid: false });

    fireEvent.click(screen.getByRole('button', { name: /View/ }));
    const reopenedViewMenu = screen.getByRole('menu', { name: 'Material graph view options' });
    fireEvent.click(within(reopenedViewMenu).getByRole('menuitem', { name: 'Stats Overlay' }));
    expect(materialState.setMaterialGraphView).toHaveBeenCalledWith(document, { showStats: true });
  });

  it('keeps reload and save-and-compile under the compile dropdown', () => {
    render(<MaterialEditorToolbar document={document} />);

    fireEvent.click(screen.getByRole('button', { name: 'Material compile actions' }));
    let actions = screen.getByRole('menu');
    fireEvent.click(within(actions).getByRole('menuitem', { name: 'Reload from Disk' }));
    expect(materialState.reloadMaterialDocument).toHaveBeenCalledWith(document);

    fireEvent.click(screen.getByRole('button', { name: 'Material compile actions' }));
    actions = screen.getByRole('menu');
    fireEvent.click(within(actions).getByRole('menuitem', { name: 'Save & Compile' }));
    expect(materialState.saveAndPublishMaterialDocument).toHaveBeenCalledWith(document);
  });
});
