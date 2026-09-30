export const editorToolbarRegions = ['left', 'center', 'right'] as const;

export type EditorToolbarRegion = (typeof editorToolbarRegions)[number];

/**
 * Stable structural contract shared by document-editor toolbars.
 * Keep domain-specific actions in their editor; this module only defines
 * the common region vocabulary used by layout, tests, and future audits.
 */
export function isEditorToolbarRegion(value: string): value is EditorToolbarRegion {
  return (editorToolbarRegions as readonly string[]).includes(value);
}
