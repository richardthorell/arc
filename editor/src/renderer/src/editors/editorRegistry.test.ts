import { FileText } from 'lucide-react';
import { afterEach, describe, expect, it } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import { getEditorDocumentsSnapshot, resetEditorDocuments } from './editorDocuments';
import { createEditorDocumentForAsset, createEditorRegistry, openAssetEditorDocument } from './editorRegistry';

const registry = createEditorRegistry({
  level: {
    kind: 'level',
    title: 'Level Editor',
    icon: FileText,
    allowMultiple: false,
    render: () => null,
    renderToolbar: () => null,
  },
});

const shader: AssetItem = {
  id: 'shader-guid',
  guid: 'shader-guid',
  name: 'pbr_lit.hlsl',
  path: 'Assets/Shaders/pbr_lit.hlsl',
  scope: 'project',
  kind: 'shader',
  status: 'ready',
  readOnly: false,
};

afterEach(resetEditorDocuments);

describe('editor registry asset routing', () => {
  it('maps shader assets to multi-document shader editors', () => {
    const target = createEditorDocumentForAsset(shader, registry);

    expect(target?.registration.kind).toBe('shader');
    expect(target?.registration.allowMultiple).toBe(true);
    expect(target?.document).toMatchObject({
      id: 'shader:shader-guid',
      kind: 'shader',
      title: 'pbr_lit.hlsl',
      path: 'Assets/Shaders/pbr_lit.hlsl',
      assetGuid: 'shader-guid',
      assetScope: 'project',
      dirty: false,
    });
  });

  it('keeps built-in shaders scoped to engine assets and read-only', () => {
    const target = createEditorDocumentForAsset(
      {
        ...shader,
        id: 'builtin-guid',
        guid: 'builtin-guid',
        name: 'default_unlit.frag',
        path: 'builtin/shaders/default_unlit.frag',
        scope: 'builtin',
        readOnly: false,
      },
      registry,
    );

    expect(target?.document).toMatchObject({
      id: 'shader:builtin-guid',
      path: 'builtin/shaders/default_unlit.frag',
      assetScope: 'builtin',
      readOnly: true,
    });
  });

  it('opens the same shader asset only once and activates it again', () => {
    expect(openAssetEditorDocument(shader, registry)).toBe(true);
    expect(openAssetEditorDocument(shader, registry)).toBe(true);

    const state = getEditorDocumentsSnapshot();
    expect(state.documents).toHaveLength(1);
    expect(state.activeDocumentId).toBe('shader:shader-guid');
  });
});
