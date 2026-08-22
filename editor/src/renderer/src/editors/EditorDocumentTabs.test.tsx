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
    expect(tab).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByLabelText('Unsaved changes')).toBeInTheDocument();
    expect(tab.querySelector('svg')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Close World\.arcscene/ })).not.toBeInTheDocument();

    fireEvent.click(tab);
    expect(onActivate).toHaveBeenCalledWith('level:world');
  });

  it('renders shader documents as closeable code-document tabs', () => {
    const onActivate = vi.fn();
    const onClose = vi.fn();
    render(
      <EditorDocumentTabs
        documents={[
          {
            id: 'shader:pbr',
            kind: 'shader',
            title: 'pbr_lit.hlsl',
            path: 'Assets/Shaders/pbr_lit.hlsl',
            assetGuid: 'pbr',
            dirty: false,
            readOnly: false,
          },
        ]}
        activeDocumentId="shader:pbr"
        registry={registry}
        onActivate={onActivate}
        onClose={onClose}
      />,
    );

    expect(screen.getByRole('tab', { name: /pbr_lit\.hlsl/ }).querySelector('svg')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close pbr_lit.hlsl' }));
    expect(onClose).toHaveBeenCalledWith('shader:pbr');
  });

  it('prompts before closing a dirty shader document', () => {
    render(
      <EditorDocumentTabs
        documents={[
          {
            id: 'shader:pbr',
            kind: 'shader',
            title: 'pbr_lit.hlsl',
            path: 'Assets/Shaders/pbr_lit.hlsl',
            assetGuid: 'pbr',
            dirty: true,
            readOnly: false,
          },
        ]}
        activeDocumentId="shader:pbr"
        registry={registry}
        onActivate={() => undefined}
        onClose={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Close pbr_lit.hlsl' }));
    expect(screen.getByRole('dialog', { name: 'Save changes?' })).toBeInTheDocument();
    expect(screen.getByRole('dialog', { name: 'Save changes?' })).toHaveTextContent(
      'pbr_lit.hlsl has unsaved changes.',
    );
  });
});
