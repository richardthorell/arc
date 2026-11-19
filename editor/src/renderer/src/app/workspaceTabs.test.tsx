// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { FileText } from 'lucide-react';
import { afterEach, describe, expect, it } from 'vitest';

import { createEditorRegistry } from '../editors/editorRegistry';
import {
  activateEditorDocumentInStore,
  getEditorDocumentsSnapshot,
  openEditorDocumentInStore,
  resetEditorDocuments,
  useEditorDocuments,
} from '../editors/editorDocuments';
import type { EditorDocument } from '../editors/editorTypes';
import { WorkspaceTabs } from '../editors/WorkspaceTabs';
import { WorkspaceToolbarHost } from '../editors/WorkspaceToolbarHost';
import {
  closeWorkspaceView,
  openWorkspaceView,
  resetWorkspaceViews,
  useWorkspaceTabs,
  viewTabId,
} from './workspaceTabs';

const registry = createEditorRegistry({
  level: {
    kind: 'level',
    title: 'Level Editor',
    icon: FileText,
    allowMultiple: false,
    render: () => null,
    renderToolbar: () => null,
  },
});

const scene: EditorDocument = {
  id: 'level:world',
  kind: 'level',
  title: 'World',
  dirty: false,
  readOnly: false,
};

function Harness() {
  const { documents, activeDocumentId, activeDocument, activateDocument, closeDocument } = useEditorDocuments();
  const { tabs, activeTabId, activeView, activateTab } = useWorkspaceTabs(documents, activeDocumentId);
  return (
    <>
      <WorkspaceTabs
        tabs={tabs}
        activeTabId={activeTabId}
        registry={registry}
        onActivate={(id) => activateTab(id, activateDocument)}
        onClose={(id) => {
          if (id.startsWith('view:')) closeWorkspaceView(id);
          else closeDocument(id);
        }}
      />
      <WorkspaceToolbarHost document={activeDocument} activeView={activeView} registry={registry} />
    </>
  );
}

afterEach(() => {
  cleanup();
  resetWorkspaceViews();
  resetEditorDocuments();
});

describe('generic workspace tab model', () => {
  it('keeps a toolbar grid slot even when Builds is the only open view', () => {
    const view = {
      type: 'view' as const,
      id: viewTabId('builds'),
      viewId: 'builds',
      title: 'Builds',
      closeable: true,
    };
    const result = render(<WorkspaceToolbarHost document={null} activeView={view} registry={registry} />);
    expect(result.container.firstElementChild).toHaveClass('editor-toolbar-host', 'editor-toolbar-host-empty');
  });

  it('opens Builds once without closing or mutating the level asset tab', () => {
    openEditorDocumentInStore(scene, false);
    render(<Harness />);
    act(() => {
      openWorkspaceView('builds', 'Builds');
      openWorkspaceView('builds', 'Builds');
    });

    expect(screen.getAllByRole('tab')).toHaveLength(2);
    const builds = screen.getByRole('tab', { name: 'Builds' });
    expect(builds).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'World' })).toHaveAttribute('aria-selected', 'false');
    // Even though the Builds view has no asset toolbar, the workbench grid
    // must retain its toolbar element or the status bar moves above content.
    expect(document.querySelector('.editor-toolbar-host')).toHaveClass('editor-toolbar-host-empty');

    fireEvent.click(screen.getByRole('button', { name: 'Close Builds' }));
    expect(screen.queryByRole('tab', { name: 'Builds' })).not.toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'World' })).toHaveAttribute('aria-selected', 'true');
    expect(document.querySelector('.editor-toolbar-host')).toHaveAttribute('data-editor-kind', 'level');
    expect(getEditorDocumentsSnapshot().documents).toEqual([scene]);
  });

  it('reuses asset close rules while tool views remain independent', () => {
    openEditorDocumentInStore(scene, false);
    openEditorDocumentInStore({
      id: 'shader:dirty',
      kind: 'shader',
      title: 'modified.hlsl',
      dirty: true,
      readOnly: false,
    });
    openWorkspaceView('builds', 'Builds');
    render(<Harness />);

    fireEvent.click(screen.getByRole('tab', { name: /modified\.hlsl/ }));
    expect(screen.getByRole('tab', { name: /modified\.hlsl/ })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('tab', { name: 'Builds' })).toHaveAttribute('aria-selected', 'false');

    fireEvent.click(screen.getByRole('button', { name: 'Close modified.hlsl' }));
    expect(screen.getByRole('dialog', { name: 'Save changes?' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Builds' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));

    fireEvent.click(screen.getByRole('tab', { name: 'Builds' }));
    expect(screen.getByRole('tab', { name: 'Builds' })).toHaveAttribute('aria-selected', 'true');
    act(() => activateEditorDocumentInStore('shader:dirty'));
    expect(screen.getByRole('tab', { name: /modified\.hlsl/ })).toHaveAttribute('aria-selected', 'true');
  });

  it('supports additional non-asset information views by ID', () => {
    openEditorDocumentInStore(scene, false);
    openWorkspaceView('builds', 'Builds');
    openWorkspaceView('analysis', 'Analysis', false);
    render(<Harness />);
    expect(screen.getAllByRole('tab')).toHaveLength(3);
    expect(screen.queryByRole('button', { name: 'Close Analysis' })).not.toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Analysis' })).toHaveAttribute('aria-selected', 'true');
    act(() => closeWorkspaceView(viewTabId('analysis')));
    expect(screen.getByRole('tab', { name: 'Analysis' })).toHaveAttribute('aria-selected', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Close Builds' }));
    expect(screen.getByRole('tab', { name: 'Analysis' })).toHaveAttribute('aria-selected', 'true');
  });
});
