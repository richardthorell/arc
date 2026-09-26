import { ChevronDown, ChevronRight } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { DragEvent, KeyboardEvent, MouseEvent, ReactNode } from 'react';

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
  selectedIds?: ReadonlySet<string>;
  defaultExpandedIds?: readonly string[];
  expandedIds?: ReadonlySet<string>;
  query?: string;
  ariaLabel?: string;
  onExpandedChange?: (expandedIds: ReadonlySet<string>) => void;
  onSelect?: (node: UiTreeNode) => void;
  onSelectionChange?: (selectedIds: ReadonlySet<string>, primaryNode: UiTreeNode) => void;
  canReparent?: (sourceIds: readonly string[], target: UiTreeNode) => boolean;
  onReparent?: (sourceIds: readonly string[], target: UiTreeNode) => void;
};

const normalize = (value: string) => value.trim().toLocaleLowerCase();

const nodeMatches = (node: UiTreeNode, query: string) => {
  if (!query) return true;
  if (normalize(node.label).includes(query)) return true;
  return node.keywords?.some((keyword) => normalize(keyword).includes(query)) ?? false;
};

const filterTree = (nodes: readonly UiTreeNode[], query: string): UiTreeNode[] => {
  if (!query)
    return nodes.map((node) => ({ ...node, children: node.children ? filterTree(node.children, '') : undefined }));

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

const findNode = (nodes: readonly UiTreeNode[], id: string): UiTreeNode | null => {
  for (const node of nodes) {
    if (node.id === id) return node;
    const child = node.children ? findNode(node.children, id) : null;
    if (child) return child;
  }
  return null;
};

const containsNode = (node: UiTreeNode, id: string): boolean =>
  node.children?.some((child) => child.id === id || containsNode(child, id)) ?? false;

export function UiTreeView({
  nodes,
  selectedId = null,
  selectedIds,
  defaultExpandedIds = [],
  expandedIds: controlledExpandedIds,
  query = '',
  ariaLabel = 'Tree',
  onExpandedChange,
  onSelect,
  onSelectionChange,
  canReparent,
  onReparent,
}: UiTreeViewProps) {
  const [uncontrolledExpandedIds, setUncontrolledExpandedIds] = useState<ReadonlySet<string>>(
    () => new Set(defaultExpandedIds),
  );
  const [focusedId, setFocusedId] = useState<string | null>(selectedId);
  const [selectionAnchorId, setSelectionAnchorId] = useState<string | null>(selectedId);
  const [draggingIds, setDraggingIds] = useState<readonly string[]>([]);
  const [dropTargetId, setDropTargetId] = useState<string | null>(null);
  const expandedIds = controlledExpandedIds ?? uncontrolledExpandedIds;
  const normalizedQuery = normalize(query);
  const effectiveSelectedIds = selectedIds ?? (selectedId ? new Set([selectedId]) : new Set<string>());

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

  const selectNode = (node: UiTreeNode, options?: { additive?: boolean; range?: boolean }) => {
    if (node.disabled) return;
    const next = new Set<string>();
    if (options?.range && selectionAnchorId) {
      const anchorIndex = visibleNodes.findIndex(({ node: candidate }) => candidate.id === selectionAnchorId);
      const targetIndex = visibleNodes.findIndex(({ node: candidate }) => candidate.id === node.id);
      if (anchorIndex >= 0 && targetIndex >= 0) {
        const [start, end] = anchorIndex <= targetIndex ? [anchorIndex, targetIndex] : [targetIndex, anchorIndex];
        if (options.additive) effectiveSelectedIds.forEach((id) => next.add(id));
        visibleNodes.slice(start, end + 1).forEach(({ node: candidate }) => {
          if (!candidate.disabled) next.add(candidate.id);
        });
      }
    } else if (options?.additive) {
      effectiveSelectedIds.forEach((id) => next.add(id));
      if (next.has(node.id)) next.delete(node.id);
      else next.add(node.id);
      setSelectionAnchorId(node.id);
    } else {
      next.add(node.id);
      setSelectionAnchorId(node.id);
    }
    if (next.size === 0 && !options?.additive) next.add(node.id);
    onSelectionChange?.(next, node);
    onSelect?.(node);
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
      selectNode(entry.node, { additive: event.ctrlKey || event.metaKey, range: event.shiftKey });
    }
  };

  const handleClick = (event: MouseEvent<HTMLButtonElement>, entry: VisibleTreeNode) => {
    setFocusedId(entry.node.id);
    selectNode(entry.node, { additive: event.ctrlKey || event.metaKey, range: event.shiftKey });
  };

  const sourceIdsFor = (id: string) =>
    effectiveSelectedIds.has(id) ? [...effectiveSelectedIds] : [id];

  const canDropOn = (sourceIds: readonly string[], target: UiTreeNode) => {
    if (target.disabled || sourceIds.includes(target.id)) return false;
    for (const sourceId of sourceIds) {
      const source = findNode(nodes, sourceId);
      if (!source || containsNode(source, target.id)) return false;
    }
    return canReparent?.(sourceIds, target) ?? true;
  };

  const handleDragStart = (event: DragEvent<HTMLButtonElement>, node: UiTreeNode) => {
    const sourceIds = sourceIdsFor(node.id);
    setDraggingIds(sourceIds);
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('application/x-arc-tree-nodes', JSON.stringify(sourceIds));
  };

  const handleDragOver = (event: DragEvent<HTMLButtonElement>, target: UiTreeNode) => {
    if (!draggingIds.length || !canDropOn(draggingIds, target)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    setDropTargetId(target.id);
  };

  const handleDrop = (event: DragEvent<HTMLButtonElement>, target: UiTreeNode) => {
    event.preventDefault();
    if (draggingIds.length && canDropOn(draggingIds, target)) onReparent?.(draggingIds, target);
    setDraggingIds([]);
    setDropTargetId(null);
  };

  const endDrag = () => {
    setDraggingIds([]);
    setDropTargetId(null);
  };

  if (visibleNodes.length === 0) return <div className="ui-tree-view-empty">No matching items</div>;

  return (
    <div
      aria-label={ariaLabel}
      aria-multiselectable={selectedIds ? true : undefined}
      className="ui-tree-view"
      role="tree"
    >
      {visibleNodes.map((entry) => {
        const hasChildren = Boolean(entry.node.children?.length);
        const expanded = hasChildren && (Boolean(normalizedQuery) || expandedIds.has(entry.node.id));
        const selected = effectiveSelectedIds.has(entry.node.id);
        return (
          <UiTreeRow
            aria-disabled={entry.node.disabled || undefined}
            aria-expanded={hasChildren ? expanded : undefined}
            aria-level={entry.depth + 1}
            aria-selected={selected}
            className={`ui-tree-view-row${dropTargetId === entry.node.id ? ' is-drop-target' : ''}`}
            data-ui-tree-node-id={entry.node.id}
            depth={entry.depth}
            disabled={entry.node.disabled}
            draggable={Boolean(onReparent) && !entry.node.disabled}
            key={entry.node.id}
            onClick={(event) => handleClick(event, entry)}
            onDoubleClick={() => toggle(entry.node)}
            onDragEnd={endDrag}
            onDragLeave={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropTargetId(null);
            }}
            onDragOver={(event) => handleDragOver(event, entry.node)}
            onDragStart={(event) => handleDragStart(event, entry.node)}
            onDrop={(event) => handleDrop(event, entry.node)}
            onKeyDown={(event) => handleKeyDown(event, entry)}
            role="treeitem"
            selected={selected}
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
