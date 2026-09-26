// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';

import {
  defaultHierarchyPanelHeight,
  defaultSceneRightColumnWidth,
  editorWorkspaceStorageKey,
  supportsRequestedWorkspacePanel,
  usesDocumentOwnedWorkspace,
} from './WorkspaceDock';

describe('WorkspaceDock document workspaces', () => {
  it('treats Flow as a document-owned workspace', () => {
    expect(usesDocumentOwnedWorkspace('flow')).toBe(true);
    expect(usesDocumentOwnedWorkspace('level')).toBe(false);
  });

  it('uses a new workspace key so existing scene layouts pick up the new defaults', () => {
    expect(editorWorkspaceStorageKey('project', 'level')).toBe('arc.editor.workspace.v8.project.editor-level');
    expect(editorWorkspaceStorageKey('project', 'flow')).toBe('arc.editor.workspace.v8.project.editor-flow-v2');
  });

  it('uses the wider stacked scene-details proportions by default', () => {
    expect(defaultSceneRightColumnWidth).toBe(560);
    expect(defaultHierarchyPanelHeight).toBe(360);
  });

  it('keeps scene-only dock panels out of Flow while preserving global utilities', () => {
    expect(supportsRequestedWorkspacePanel('flow', 'hierarchy')).toBe(false);
    expect(supportsRequestedWorkspacePanel('flow', 'inspector')).toBe(false);
    expect(supportsRequestedWorkspacePanel('flow', 'contentBrowser')).toBe(false);
    expect(supportsRequestedWorkspacePanel('flow', 'search')).toBe(true);
    expect(supportsRequestedWorkspacePanel('flow', 'viewport')).toBe(true);
    expect(supportsRequestedWorkspacePanel('level', 'hierarchy')).toBe(true);
  });
});
