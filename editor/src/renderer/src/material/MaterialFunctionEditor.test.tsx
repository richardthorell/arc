// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { MaterialFunctionEditor } from './MaterialFunctionEditor';
import { createDefaultMaterialFunction } from './materialGraphTypes';

const functionState = vi.hoisted(() => ({
  redoMaterialFunctionGraph: vi.fn(),
  replaceMaterialFunctionAsset: vi.fn(),
  replaceMaterialFunctionGraph: vi.fn(),
  replaceMaterialFunctionViewport: vi.fn(),
  undoMaterialFunctionGraph: vi.fn(),
  useMaterialFunctionDocumentState: vi.fn(),
}));

vi.mock('./materialFunctionDocumentState', () => functionState);

vi.mock('./MaterialGraphInteractions', () => ({
  MaterialGraphWithInteractions: () => <div data-testid="shared-material-graph">Shared material graph</div>,
}));

const document: EditorDocument = {
  id: 'function-editor-test',
  kind: 'materialFunction',
  title: 'Tint Function',
  path: 'Content/Tint.arcmatfn',
  assetGuid: '00112233445566778899aabbccddeeff',
  assetScope: 'project',
  dirty: false,
  readOnly: false,
};

beforeEach(() => {
  const asset = createDefaultMaterialFunction('Tint Function');
  functionState.useMaterialFunctionDocumentState.mockReturnValue({
    documentId: document.id,
    path: document.path,
    readOnly: false,
    asset,
    graph: asset.graph,
    confirmed: '',
    history: [asset.graph],
    historyIndex: 0,
    loaded: true,
    loading: false,
    saving: false,
    validating: false,
    message: '',
  });
});

describe('MaterialFunctionEditor', () => {
  it('uses the shared material graph workspace without material preview/output settings', () => {
    const { container } = render(<MaterialFunctionEditor document={document} />);

    expect(container.querySelector('.material-graph-workspace')).toBeInTheDocument();
    expect(screen.getByTestId('shared-material-graph')).toBeInTheDocument();
    expect(screen.getByText('Material Function')).toBeInTheDocument();
    expect(screen.getByText('Inputs')).toBeInTheDocument();
    expect(screen.getByText('Outputs')).toBeInTheDocument();

    expect(screen.queryByText('Material Preview')).not.toBeInTheDocument();
    expect(screen.queryByText('Rendering')).not.toBeInTheDocument();
    expect(screen.queryByText('Masking')).not.toBeInTheDocument();
    expect(screen.queryByText('Translucency')).not.toBeInTheDocument();
    expect(screen.queryByText('Advanced')).not.toBeInTheDocument();
  });
});
