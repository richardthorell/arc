import type { HTMLAttributes } from 'react';

import { UiSearchInput } from './UiTextInput';
import { UiTreeView } from './UiTreeView';
import type { UiTreeNode } from './UiTreeView';

import './UiSettingsNavigation.css';

export type UiSettingsNavigationProps = Omit<HTMLAttributes<HTMLDivElement>, 'onSelect'> & {
  nodes: readonly UiTreeNode[];
  selectedId?: string | null;
  query: string;
  defaultExpandedIds?: readonly string[];
  searchAriaLabel?: string;
  searchPlaceholder?: string;
  treeAriaLabel?: string;
  onQueryChange: (query: string) => void;
  onSelect?: (node: UiTreeNode) => void;
};

export function UiSettingsNavigation({
  nodes,
  selectedId = null,
  query,
  defaultExpandedIds = [],
  searchAriaLabel = 'Search settings',
  searchPlaceholder = 'Search settings',
  treeAriaLabel = 'Settings sections',
  onQueryChange,
  onSelect,
  className,
  ...props
}: UiSettingsNavigationProps) {
  return (
    <div className={['ui-settings-navigation', className].filter(Boolean).join(' ')} {...props}>
      <UiSearchInput
        aria-label={searchAriaLabel}
        autoFocus={false}
        onChange={(event) => onQueryChange(event.target.value)}
        placeholder={searchPlaceholder}
        value={query}
      />
      <UiTreeView
        ariaLabel={treeAriaLabel}
        defaultExpandedIds={defaultExpandedIds}
        nodes={nodes}
        onSelect={onSelect}
        query={query}
        selectedId={selectedId}
      />
    </div>
  );
}
