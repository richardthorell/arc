// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ExplorerPanel } from './Workbench';
import type { ProjectSnapshot } from '../services/editorHostTypes';

afterEach(() => document.body.replaceChildren());

describe('ExplorerPanel', () => {
  it('renders the entity palette as an in-panel drawer', () => {
    const onCreateEntity = vi.fn();
    const project = { scene: [] } as unknown as ProjectSnapshot;
    const view = render(
      <ExplorerPanel
        project={project}
        selectedEntityId=""
        selectedEntityIds={new Set()}
        onSelectEntity={vi.fn()}
        onRenameEntity={vi.fn()}
        onSetEntityActive={vi.fn()}
        onMoveEntity={vi.fn()}
        onCreateEntity={onCreateEntity}
        onDuplicate={vi.fn()}
        onCreatePrefab={vi.fn()}
        onInstantiatePrefab={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Add entity' }));
    const palette = screen.getByRole('menu', { name: 'Add entity' });
    expect(view.container.contains(palette)).toBe(true);
    expect(
      palette.compareDocumentPosition(screen.getByRole('textbox', { name: 'Search hierarchy' })) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    fireEvent.click(screen.getByRole('menuitem', { name: 'Box' }));
    expect(onCreateEntity).toHaveBeenCalledWith('cube');
    expect(screen.queryByRole('menu', { name: 'Add entity' })).not.toBeInTheDocument();
  });
});
