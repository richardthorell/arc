export type HierarchyContextSelection = Readonly<{
  selectedIds: readonly string[];
  primaryId: string | null;
}>;

export type HierarchyContextActionId =
  | 'hierarchy.rename'
  | 'hierarchy.duplicate'
  | 'hierarchy.delete'
  | 'hierarchy.create-child';

export type HierarchyContextAction = Readonly<{
  id: HierarchyContextActionId;
  label: string;
  enabled: boolean;
  destructive?: boolean;
}>;

/**
 * Builds the entity actions shared by pointer and keyboard-invoked Hierarchy
 * context menus. Keeping stable IDs and enablement outside the menu surface
 * lets the panel route every entry through the same command/undo path.
 */
export function buildHierarchyContextActions(selection: HierarchyContextSelection): readonly HierarchyContextAction[] {
  const selectionCount = selection.selectedIds.length;
  const hasSelection = selectionCount > 0;
  const hasPrimary = selection.primaryId != null && selection.selectedIds.includes(selection.primaryId);
  const singleSelection = selectionCount === 1 && hasPrimary;

  return [
    { id: 'hierarchy.rename', label: 'Rename', enabled: singleSelection },
    {
      id: 'hierarchy.duplicate',
      label: selectionCount > 1 ? `Duplicate ${selectionCount} Entities` : 'Duplicate',
      enabled: hasSelection,
    },
    { id: 'hierarchy.create-child', label: 'Create Child', enabled: singleSelection },
    {
      id: 'hierarchy.delete',
      label: selectionCount > 1 ? `Delete ${selectionCount} Entities` : 'Delete',
      enabled: hasSelection,
      destructive: true,
    },
  ];
}
