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

const sound: AssetItem = {
  id: 'sound-guid',
  guid: 'sound-guid',
  name: 'Footstep.arcsound',
  path: 'Audio/Footstep.arcsound',
  sourcePath: 'Content/Audio/Footstep.arcsound',
  scope: 'project',
  kind: 'sound',
  status: 'ready',
  readOnly: false,
};

describe('Sound editor registration', () => {
  it('routes .arcsound assets to independent Sound workspace documents', () => {
    const target = createEditorDocumentForAsset(sound, registry);

    expect(target?.registration.kind).toBe('sound');
    expect(target?.registration.allowMultiple).toBe(true);
    expect(target?.registration.save).toBeTypeOf('function');
    expect(target?.registration.onClosed).toBeTypeOf('function');
    expect(target?.document).toMatchObject({
      id: 'sound:sound-guid',
      kind: 'sound',
      title: 'Footstep.arcsound',
      path: 'Content/Audio/Footstep.arcsound',
      assetGuid: 'sound-guid',
      assetScope: 'project',
      dirty: false,
      readOnly: false,
    });
  });

  it('also recognizes authored .arcsound paths while registry metadata catches up', () => {
    const target = createEditorDocumentForAsset({ ...sound, kind: 'unknown', path: 'Audio/New Sound.arcsound' }, registry);
    expect(target?.registration.kind).toBe('sound');
  });
});
