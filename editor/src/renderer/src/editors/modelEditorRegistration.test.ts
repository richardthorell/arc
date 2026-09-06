// @vitest-environment jsdom
import { FileText } from 'lucide-react';
import { describe, expect, it } from 'vitest';
import type { AssetItem } from '../services/editorHostTypes';
import { createEditorDocumentForAsset, createEditorRegistry } from './editorRegistry';

const registry = createEditorRegistry({
  level: {
    kind: 'level',
    title: 'Level',
    icon: FileText,
    allowMultiple: false,
    render: () => null,
    renderToolbar: () => null,
  },
});
const fbx: AssetItem = {
  id: 'hand',
  guid: 'hand',
  name: 'Hand.fbx',
  path: 'Content/Hand.fbx',
  kind: 'scene',
  status: 'ready',
  meshCount: 1,
  skeletonBoneCount: 27,
};

describe('model editor registration', () => {
  it('opens FBX models in the model editor while keeping skeletons as sub-assets', () => {
    const target = createEditorDocumentForAsset(fbx, registry);
    expect(target?.document).toMatchObject({ id: 'model:hand', kind: 'model', assetSnapshot: fbx });
    expect(target?.registration.kind).toBe('model');
  });
});
