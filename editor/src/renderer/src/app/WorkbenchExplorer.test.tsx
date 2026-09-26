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

  it('exposes World as a first-class hierarchy inspection target', () => {
    const onSelectEntity = vi.fn();
    render(
      <ExplorerPanel
        project={{ scene: [] } as unknown as ProjectSnapshot}
        selectedEntityId=""
        selectedEntityIds={new Set()}
        onSelectEntity={onSelectEntity}
        onRenameEntity={vi.fn()}
        onSetEntityActive={vi.fn()}
        onMoveEntity={vi.fn()}
        onCreateEntity={vi.fn()}
        onDuplicate={vi.fn()}
        onCreatePrefab={vi.fn()}
        onInstantiatePrefab={vi.fn()}
        onDelete={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole('treeitem', { name: 'World' }));
    expect(onSelectEntity).toHaveBeenCalledWith('world');
  });

  it('routes Terrain through the dedicated authoring workflow', () => {
    const onCreateEntity = vi.fn();
    render(
      <ExplorerPanel
        project={{ scene: [] } as unknown as ProjectSnapshot}
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
    fireEvent.click(screen.getByRole('menuitem', { name: 'Terrain...' }));
    expect(onCreateEntity).toHaveBeenCalledWith('terrain');
  });

  it('labels Play World hierarchy data and disables authoring actions', () => {
    const onSelectEntity = vi.fn();
    const project = {
      scene: [{ id: '7:2', guid: 'runtime-guid', name: 'Spawned Actor', kind: 'mesh', active: true, children: [] }],
    } as unknown as ProjectSnapshot;
    render(
      <ExplorerPanel
        project={project}
        selectedEntityId=""
        selectedEntityIds={new Set()}
        onSelectEntity={onSelectEntity}
        onRenameEntity={vi.fn()}
        onSetEntityActive={vi.fn()}
        onMoveEntity={vi.fn()}
        onCreateEntity={vi.fn()}
        onDuplicate={vi.fn()}
        onCreatePrefab={vi.fn()}
        onInstantiatePrefab={vi.fn()}
        onDelete={vi.fn()}
        readOnly
        worldLabel="Play World"
      />,
    );

    expect(screen.getByText('Play World')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add entity' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Delete selected entity' })).toBeDisabled();
    fireEvent.click(screen.getByText('Spawned Actor'));
    expect(onSelectEntity).toHaveBeenCalledWith('7:2', false);
  });
});
