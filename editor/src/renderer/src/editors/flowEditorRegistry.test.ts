// @vitest-environment jsdom
import { FileText } from 'lucide-react';
import { afterEach, describe, expect, it } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import { resetEditorDocuments } from './editorDocuments';
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

const flowAsset: AssetItem = {
  id: 'flow-guid',
  guid: 'flow-guid',
  name: 'PlayerController.arcflow',
  path: 'Content/Logic/PlayerController.arcflow',
  scope: 'project',
  kind: 'flow',
  status: 'ready',
  readOnly: false,
};

afterEach(resetEditorDocuments);

describe('Flow editor registry routing', () => {
  it('maps Flow assets to independent Flow editor documents', () => {
    const target = createEditorDocumentForAsset(flowAsset, registry);

    expect(target?.registration).toMatchObject({ kind: 'flow', title: 'Flow Graph Editor', allowMultiple: true });
    expect(target?.document).toMatchObject({
      id: 'flow:flow-guid',
      kind: 'flow',
      title: 'PlayerController.arcflow',
      path: 'Content/Logic/PlayerController.arcflow',
      assetGuid: 'flow-guid',
      assetScope: 'project',
      dirty: false,
      readOnly: false,
    });
  });

  it('recognizes .arcflow files even when native discovery reports them as unknown', () => {
    const target = createEditorDocumentForAsset(
      {
        ...flowAsset,
        id: 'Content/Logic/Legacy.arcflow',
        guid: undefined,
        name: 'Legacy.arcflow',
        path: 'Content/Logic/Legacy.arcflow',
        kind: 'unknown',
      },
      registry,
    );

    expect(target?.registration.kind).toBe('flow');
    expect(target?.document.kind).toBe('flow');
    expect(target?.document.id).toBe('flow:Content/Logic/Legacy.arcflow');
  });
});
