import type { EditorToolbarRegion } from './editorToolbarContract';

export type EditorToolbarPrimaryAction = 'save' | 'build' | 'compile';

export type EditorToolbarActionPlacement = Readonly<{
  region: EditorToolbarRegion;
  supportsMenu: boolean;
}>;

/**
 * Canonical placement/interaction contract for primary editor actions.
 *
 * Document-local authoring actions stay on the left so Save/Compile read
 * consistently across asset editors. Scene-level Build stays on the right
 * with target platform/device controls. Domains own labels, disabled state,
 * and menu entries; this contract only owns shared toolbar semantics.
 */
export const editorToolbarPrimaryActions: Readonly<Record<EditorToolbarPrimaryAction, EditorToolbarActionPlacement>> = {
  save: { region: 'left', supportsMenu: true },
  compile: { region: 'left', supportsMenu: true },
  build: { region: 'right', supportsMenu: true },
};

export function editorToolbarPlacementFor(action: EditorToolbarPrimaryAction): EditorToolbarActionPlacement {
  return editorToolbarPrimaryActions[action];
}
