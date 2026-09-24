// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';

import {
  editorWorkspaceStorageKey,
  supportsRequestedWorkspacePanel,
  usesDocumentOwnedWorkspace,
} from './WorkspaceDock';

describe('WorkspaceDock document workspaces', () => {
  it('treats Flow as a document-owned workspace', () => {
    expect(usesDocumentOwnedWorkspace('flow')).toBe(true);
    expect(usesDocumentOwnedWorkspace('level')).toBe(false);
  });

  it('uses a new Flow workspace key so old Level-style layouts are not restored', () => {
    expect(editorWorkspaceStorageKey('project', 'flow')).toBe('arc.editor.workspace.v7.project.editor-flow-v2');
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
