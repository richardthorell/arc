import { ChevronDown, ChevronRight } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';

import { UiTreeRow } from './UiTreeRow';
import './UiTreeView.css';

export type UiTreeNode = {
  id: string;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
  keywords?: readonly string[];
  children?: readonly UiTreeNode[];
};

type VisibleTreeNode = {
  node: UiTreeNode;
  depth: number;
  parentId: string | null;
};

type UiTreeViewProps = {
  nodes: readonly UiTreeNode[];
  selectedId?: string | null;
  defaultExpandedIds?: readonly string[];
  expandedIds?: ReadonlySet<string>;
  query?: string;
  ariaLabel?: string;
  onExpandedChange?: (expandedIds: ReadonlySet<string>) => void;
  onSelect?: (node: UiTreeNode) => void;
};

const normalize = (value: string) => value.trim().toLocaleLowerCase();

const nodeMatches = (node: UiTreeNode, query: string) => {
  if (!query) return true;
  if (normalize(node.label).includes(query)) return true;
  return node.keywords?.some((keyword) => normalize(keyword).includes(query)) ?? false;
};

const filterTree = (nodes: readonly UiTreeNode[], query: string): UiTreeNode[] => {
  if (!query) return nodes.map((node) => ({ ...node, children: node.children ? filterTree(node.children, '') : undefined }));

  return nodes.flatMap((node) => {
    const children = node.children ? filterTree(node.children, query) : [];
    if (!nodeMatches(node, query) && children.length === 0) return [];
    return [{ ...node, children }];
  });
};

const flattenVisibleNodes = (
  nodes: readonly UiTreeNode[],
  expandedIds: ReadonlySet<string>,
  forceExpanded: boolean,
  depth = 0,
  parentId: string | null = null,
): VisibleTreeNode[] => {
  const result: VisibleTreeNode[] = [];
  for (const node of nodes) {
    result.push({ node, depth, parentId });
    const expanded = forceExpanded || expandedIds.has(node.id);
    if (expanded && node.children?.length)
      result.push(...flattenVisibleNodes(node.children, expandedIds, forceExpanded, depth + 1, node.id));
  }
  return result;
};

export function UiTreeView({
  nodes,
  selectedId = null,
  defaultExpandedIds = [],
  expandedIds: controlledExpandedIds,
  query = '',
  ariaLabel = 'Tree',
  onExpandedChange,
  onSelect,
}: UiTreeViewProps) {
  const [uncontrolledExpandedIds, setUncontrolledExpandedIds] = useState<ReadonlySet<string>>(
    () => new Set(defaultExpandedIds),
  );
  const [focusedId, setFocusedId] = useState<string | null>(selectedId);
  const expandedIds = controlledExpandedIds ?? uncontrolledExpandedIds;
  const normalizedQuery = normalize(query);

  const filteredNodes = useMemo(() => filterTree(nodes, normalizedQuery), [nodes, normalizedQuery]);
  const visibleNodes = useMemo(
    () => flattenVisibleNodes(filteredNodes, expandedIds, Boolean(normalizedQuery)),
    [expandedIds, filteredNodes, normalizedQuery],
  );

  useEffect(() => {
    if (selectedId && visibleNodes.some(({ node }) => node.id === selectedId)) setFocusedId(selectedId);
  }, [selectedId, visibleNodes]);

  useEffect(() => {
    if (focusedId && visibleNodes.some(({ node }) => node.id === focusedId)) return;
    setFocusedId(visibleNodes[0]?.node.id ?? null);
  }, [focusedId, visibleNodes]);

  const setExpandedIds = (next: ReadonlySet<string>) => {
    if (!controlledExpandedIds) setUncontrolledExpandedIds(next);
    onExpandedChange?.(next);
  };

  const toggle = (node: UiTreeNode, expanded?: boolean) => {
    if (!node.children?.length) return;
    const next = new Set(expandedIds);
    const shouldExpand = expanded ?? !next.has(node.id);
    if (shouldExpand) next.add(node.id);
    else next.delete(node.id);
    setExpandedIds(next);
  };

  const focusNode = (id: string | null) => {
    if (!id) return;
    setFocusedId(id);
    queueMicrotask(() => {
      const element = document.querySelector<HTMLButtonElement>(`[data-ui-tree-node-id="${CSS.escape(id)}"]`);
      element?.focus();
    });
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, entry: VisibleTreeNode) => {
    const index = visibleNodes.findIndex(({ node }) => node.id === entry.node.id);
    const children = entry.node.children ?? [];

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      focusNode(visibleNodes[Math.min(index + 1, visibleNodes.length - 1)]?.node.id ?? null);
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      focusNode(visibleNodes[Math.max(index - 1, 0)]?.node.id ?? null);
      return;
    }
    if (event.key === 'Home') {
      event.preventDefault();
      focusNode(visibleNodes[0]?.node.id ?? null);
      return;
    }
    if (event.key === 'End') {
      event.preventDefault();
      focusNode(visibleNodes[visibleNodes.length - 1]?.node.id ?? null);
      return;
    }
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      if (children.length && !expandedIds.has(entry.node.id)) toggle(entry.node, true);
      else if (children.length) focusNode(children[0]?.id ?? null);
      return;
    }
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      if (children.length && expandedIds.has(entry.node.id)) toggle(entry.node, false);
      else focusNode(entry.parentId);
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (!entry.node.disabled) onSelect?.(entry.node);
    }
  };

  if (visibleNodes.length === 0) return <div className="ui-tree-view-empty">No matching items</div>;

  return (
    <div aria-label={ariaLabel} className="ui-tree-view" role="tree">
      {visibleNodes.map((entry) => {
        const hasChildren = Boolean(entry.node.children?.length);
        const expanded = hasChildren && (Boolean(normalizedQuery) || expandedIds.has(entry.node.id));
        return (
          <UiTreeRow
            aria-disabled={entry.node.disabled || undefined}
            aria-expanded={hasChildren ? expanded : undefined}
            aria-level={entry.depth + 1}
            aria-selected={entry.node.id === selectedId}
            className="ui-tree-view-row"
            data-ui-tree-node-id={entry.node.id}
            depth={entry.depth}
            disabled={entry.node.disabled}
            key={entry.node.id}
            onClick={() => {
              setFocusedId(entry.node.id);
              if (!entry.node.disabled) onSelect?.(entry.node);
            }}
            onDoubleClick={() => toggle(entry.node)}
            onKeyDown={(event) => handleKeyDown(event, entry)}
            role="treeitem"
            selected={entry.node.id === selectedId}
            tabIndex={entry.node.id === focusedId ? 0 : -1}
          >
            <span
              aria-hidden="true"
              className={`ui-tree-view-disclosure${hasChildren ? ' is-visible' : ''}`}
              onClick={(event) => {
                if (!hasChildren) return;
                event.stopPropagation();
                toggle(entry.node);
              }}
            >
              {hasChildren ? expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} /> : null}
            </span>
            {entry.node.icon && <span className="ui-tree-view-icon">{entry.node.icon}</span>}
            <span className="ui-tree-view-label">{entry.node.label}</span>
          </UiTreeRow>
        );
      })}
    </div>
  );
}
