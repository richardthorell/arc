// @vitest-environment jsdom
import { FileText } from 'lucide-react';
import { afterEach, describe, expect, it, vi } from 'vitest';

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

const material: AssetItem = {
  id: 'material-guid',
  guid: 'material-guid',
  name: 'M_Rock.arcmat',
  path: 'Content/Materials/M_Rock.arcmat',
  scope: 'project',
  kind: 'material',
  status: 'ready',
  readOnly: false,
};

afterEach(() => {
  resetEditorDocuments();
  vi.restoreAllMocks();
  Reflect.deleteProperty(window, 'arc');
});

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

  it('maps material assets to independent material documents', () => {
    const target = createEditorDocumentForAsset(material, registry);

    expect(target?.registration.kind).toBe('material');
    expect(target?.registration.allowMultiple).toBe(true);
    expect(target?.document).toMatchObject({
      id: 'material:material-guid',
      kind: 'material',
      title: 'M_Rock.arcmat',
      path: 'Content/Materials/M_Rock.arcmat',
      assetGuid: 'material-guid',
      assetScope: 'project',
      dirty: false,
      readOnly: false,
    });
  });

  it('forces built-in materials read-only', () => {
    const target = createEditorDocumentForAsset(
      {
        ...material,
        id: 'builtin-material-guid',
        guid: 'builtin-material-guid',
        path: 'builtin/materials/default.arcmat',
        scope: 'builtin',
        readOnly: false,
      },
      registry,
    );

    expect(target?.document).toMatchObject({
      id: 'material:builtin-material-guid',
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

  it('keeps multiple material documents while de-duplicating the same asset', () => {
    expect(openAssetEditorDocument(material, registry)).toBe(true);
    expect(openAssetEditorDocument({ ...material, id: 'water', guid: 'water', name: 'M_Water.arcmat' }, registry)).toBe(
      true,
    );
    expect(openAssetEditorDocument(material, registry)).toBe(true);

    const state = getEditorDocumentsSnapshot();
    expect(state.documents.map((document) => document.id)).toEqual(['material:material-guid', 'material:water']);
    expect(state.activeDocumentId).toBe('material:material-guid');
  });

  it('waits for a newly-authored material to receive its native GUID before opening it', async () => {
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      payload: {
        projectRoot: 'D:/Project',
        assetRoot: 'D:/Project/Content',
        assets: [
          {
            guid: 'registered-material-guid',
            path: 'Materials/New Material.arcmat',
            scope: 'project',
            readOnly: false,
            state: 'ready',
          },
        ],
      },
    });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: { host: { query } },
    });

    expect(
      openAssetEditorDocument(
        {
          id: 'Content/Materials/New Material.arcmat',
          name: 'New Material.arcmat',
          path: 'Content/Materials/New Material.arcmat',
          scope: 'project',
          kind: 'material',
          status: 'ready',
        },
        registry,
      ),
    ).toBe(true);
    expect(getEditorDocumentsSnapshot().documents).toHaveLength(0);

    await vi.waitFor(() => {
      expect(getEditorDocumentsSnapshot().activeDocumentId).toBe('material:registered-material-guid');
    });
    expect(query).toHaveBeenCalledWith('project.assets');
    expect(getEditorDocumentsSnapshot().documents[0]).toMatchObject({
      assetGuid: 'registered-material-guid',
      path: 'Content/Materials/New Material.arcmat',
    });
  });

  it('canonicalizes an already-registered asset-root-relative material path before opening', async () => {
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      payload: {
        projectRoot: 'D:/Project',
        assetRoot: 'D:/Project/Content',
        assets: [
          {
            guid: 'material-guid',
            path: 'pr_test.arcmat',
            scope: 'project',
            readOnly: false,
            state: 'ready',
          },
        ],
      },
    });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: { host: { query } },
    });

    expect(openAssetEditorDocument({ ...material, path: 'pr_test.arcmat' }, registry)).toBe(true);
    expect(getEditorDocumentsSnapshot().documents).toHaveLength(0);

    await vi.waitFor(() => {
      expect(getEditorDocumentsSnapshot().activeDocumentId).toBe('material:material-guid');
    });
    expect(getEditorDocumentsSnapshot().documents[0]).toMatchObject({
      assetGuid: 'material-guid',
      path: 'Content/pr_test.arcmat',
    });
  });
});
