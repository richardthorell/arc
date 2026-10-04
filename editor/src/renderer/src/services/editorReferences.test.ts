import { describe, expect, it, vi } from 'vitest';

import {
  createEditorReferenceController,
  editorReferenceUri,
  parseEditorReference,
  type EditorReference,
} from './editorReferences';

describe('editor references', () => {
  it('parses and serializes stable ARC references', () => {
    expect(parseEditorReference('arc://entity/player-guid')).toEqual({ kind: 'entity', id: 'player-guid' });
    expect(parseEditorReference('arc://asset/material%20guid')).toEqual({ kind: 'asset', id: 'material guid' });
    expect(editorReferenceUri({ kind: 'scene', id: 'scene/main' })).toBe('arc://scene/scene%2Fmain');
  });

  it('rejects unsupported or malformed references', () => {
    expect(parseEditorReference('https://example.com')).toBeNull();
    expect(parseEditorReference('arc://component/foo')).toBeNull();
    expect(parseEditorReference('arc://entity/')).toBeNull();
    expect(parseEditorReference('arc://entity/foo/bar')).toBeNull();
  });

  it('routes resolve, activate, focus, and highlight by reference kind', async () => {
    const activateEntity = vi.fn();
    const focusEntity = vi.fn();
    const highlightEntity = vi.fn();
    const controller = createEditorReferenceController({
      resolveEntity: (id) => ({ label: `Entity ${id}`, subtitle: 'Mesh' }),
      activateEntity,
      focusEntity,
      highlightEntity,
    });
    const reference: EditorReference = { kind: 'entity', id: '42' };

    await expect(controller.resolve(reference)).resolves.toEqual({
      reference,
      label: 'Entity 42',
      subtitle: 'Mesh',
    });
    await controller.activate(reference);
    await controller.focus?.(reference);
    await controller.highlight?.(reference, true);

    expect(activateEntity).toHaveBeenCalledWith('42');
    expect(focusEntity).toHaveBeenCalledWith('42');
    expect(highlightEntity).toHaveBeenCalledWith('42', true);
  });

  it('falls back to activation when a dedicated focus handler is not provided', async () => {
    const activateAsset = vi.fn();
    const controller = createEditorReferenceController({ activateAsset });

    await controller.focus?.({ kind: 'asset', id: 'wood' });

    expect(activateAsset).toHaveBeenCalledWith('wood');
  });
});
