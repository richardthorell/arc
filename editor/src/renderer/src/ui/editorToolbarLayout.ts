export type EditorToolbarSection = 'left' | 'center' | 'right';

export interface EditorToolbarItem<T = unknown> {
  id: string;
  section: EditorToolbarSection;
  order?: number;
  value: T;
}

export interface EditorToolbarLayout<T = unknown> {
  left: EditorToolbarItem<T>[];
  center: EditorToolbarItem<T>[];
  right: EditorToolbarItem<T>[];
}

const sections: readonly EditorToolbarSection[] = ['left', 'center', 'right'];

function compareToolbarItems<T>(left: EditorToolbarItem<T>, right: EditorToolbarItem<T>): number {
  const order = (left.order ?? 0) - (right.order ?? 0);
  if (order !== 0) return order;
  return left.id.localeCompare(right.id);
}

/**
 * Builds the shared left/center/right editor-toolbar presentation contract.
 *
 * Domain editors own their actions, but not toolbar placement semantics. Stable
 * ids provide deterministic ordering when actions share an authored order and
 * make the resulting layout suitable for tests, menus, and future customization.
 */
export function buildEditorToolbarLayout<T>(items: readonly EditorToolbarItem<T>[]): EditorToolbarLayout<T> {
  const seenIds = new Set<string>();
  const layout: EditorToolbarLayout<T> = { left: [], center: [], right: [] };

  for (const item of items) {
    const id = item.id.trim();
    if (id.length === 0) throw new Error('Editor toolbar item ids must not be empty.');
    if (seenIds.has(id)) throw new Error(`Duplicate editor toolbar item id: ${id}`);
    seenIds.add(id);

    layout[item.section].push({ ...item, id });
  }

  for (const section of sections) layout[section].sort(compareToolbarItems);
  return layout;
}
