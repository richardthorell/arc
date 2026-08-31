// @vitest-environment jsdom
import { FileText } from 'lucide-react';
import { describe, expect, it } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import { createEditorDocumentForAsset, createEditorRegistry } from './editorRegistry';

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

const texture: AssetItem = {
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
};

describe('texture editor registration', () => {
  it('maps textures to independent editor documents and preserves their metadata', () => {
    const target = createEditorDocumentForAsset(texture, registry);

    expect(target?.registration.kind).toBe('texture');
    expect(target?.registration.allowMultiple).toBe(true);
    expect(target?.document).toMatchObject({
      id: 'texture:texture-guid',
      kind: 'texture',
      title: 'T_Rock.png',
      path: 'Content/Textures/T_Rock.png',
      assetGuid: 'texture-guid',
      assetScope: 'project',
      dirty: false,
      readOnly: false,
      assetSnapshot: expect.objectContaining({
        width: 2048,
        height: 1024,
        mipLevels: 12,
      }),
    });
  });
});
