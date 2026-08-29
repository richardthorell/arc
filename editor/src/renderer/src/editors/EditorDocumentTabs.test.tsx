// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { FileText } from 'lucide-react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { createEditorRegistry } from './editorRegistry';
import { EditorDocumentTabs } from './EditorDocumentTabs';

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

afterEach(cleanup);

describe('EditorDocumentTabs', () => {
  it('renders the active level document with the world icon and routes activation', () => {
    const onActivate = vi.fn();
    render(
      <EditorDocumentTabs
        documents={[
          {
            id: 'level:world',
            kind: 'level',
            title: 'World.arcscene',
            path: 'Assets/World.arcscene',
            dirty: true,
            readOnly: false,
          },
        ]}
        activeDocumentId="level:world"
        registry={registry}
        onActivate={onActivate}
      />,
    );

    const tab = screen.getByRole('tab', { name: /World\.arcscene/ });
    const icon = tab.querySelector('[data-document-type-icon="level"]');
    expect(tab).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByLabelText('Unsaved changes')).toBeInTheDocument();
    expect(icon).toBeInTheDocument();
    expect(icon).toHaveClass('editor-document-tab-icon');
    expect(tab.querySelector('.editor-document-tab-title')).toHaveTextContent('World.arcscene');
    expect(screen.queryByRole('button', { name: /Close World\.arcscene/ })).not.toBeInTheDocument();

    fireEvent.click(tab);
    expect(onActivate).toHaveBeenCalledWith('level:world');
  });

  it('renders shader documents with a visible atlas icon and close action', () => {
    const onActivate = vi.fn();
    const onClose = vi.fn();
    render(
      <EditorDocumentTabs
        documents={[
          {
            id: 'shader:test',
            kind: 'shader',
            title: 'test.hlsl',
            path: 'Assets/Shaders/test.hlsl',
            assetGuid: 'shader-test',
            dirty: false,
            readOnly: false,
          },
        ]}
        activeDocumentId="shader:test"
        registry={registry}
        onActivate={onActivate}
        onClose={onClose}
      />,
    );

    const tab = screen.getByRole('tab', { name: /test\.hlsl/ });
    expect(tab.querySelector('[data-document-type-icon="shader"]')).toHaveClass('editor-document-tab-icon');
    fireEvent.click(screen.getByRole('button', { name: 'Close test.hlsl' }));
    expect(onClose).toHaveBeenCalledWith('shader:test');
  });

  it('prompts before closing a dirty shader document', () => {
    render(
      <EditorDocumentTabs
        documents={[
          {
            id: 'shader:test',
            kind: 'shader',
            title: 'test.hlsl',
            path: 'Assets/Shaders/test.hlsl',
            assetGuid: 'shader-test',
            dirty: true,
            readOnly: false,
          },
        ]}
        activeDocumentId="shader:test"
        registry={registry}
        onActivate={() => undefined}
        onClose={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Close test.hlsl' }));
    expect(screen.getByRole('dialog', { name: 'Save changes?' })).toBeInTheDocument();
    expect(screen.getByRole('dialog', { name: 'Save changes?' })).toHaveTextContent('test.hlsl has unsaved changes.');
  });
});
