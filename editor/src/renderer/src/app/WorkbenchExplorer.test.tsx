// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ExplorerPanel } from './Workbench';
import type { ProjectSnapshot } from '../services/editorHostTypes';

afterEach(() => document.body.replaceChildren());

describe('ExplorerPanel', () => {
  it('renders entity creation as a compact dropdown context menu', () => {
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
    const menu = screen.getByRole('menu', { name: 'Add entity' });
    expect(menu).toHaveClass('ui-context-menu', 'ui-context-menu-portal', 'hierarchy-create-dropdown');
    expect(view.container.querySelector('.hierarchy-create-menu')?.contains(menu)).toBe(false);
    expect(menu.parentElement).toBe(document.body);
    expect(screen.getByRole('menuitem', { name: 'Empty Entity' })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: 'Box' })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: 'Terrain...' })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: 'Ocean' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('menuitem', { name: 'Box' }));
    expect(onCreateEntity).toHaveBeenCalledWith('cube');
    expect(screen.queryByRole('menu', { name: 'Add entity' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Add entity' }));
    fireEvent.pointerDown(screen.getByRole('textbox', { name: 'Search hierarchy' }));
    expect(screen.queryByRole('menu', { name: 'Add entity' })).not.toBeInTheDocument();
  });

  it('filters the Create Entity menu and uses Escape to clear before closing', () => {
    render(
      <ExplorerPanel
        project={{ scene: [] } as unknown as ProjectSnapshot}
        selectedEntityId=""
        selectedEntityIds={new Set()}
        onSelectEntity={vi.fn()}
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

    fireEvent.click(screen.getByRole('button', { name: 'Add entity' }));
    const search = screen.getByRole('searchbox', { name: 'Search entities' });
    expect(search).toHaveFocus();

    fireEvent.change(search, { target: { value: 'sph' } });
    expect(screen.getByRole('menuitem', { name: 'Sphere' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Box' })).not.toBeInTheDocument();

    fireEvent.keyDown(search, { key: 'Escape' });
    expect(screen.getByRole('searchbox', { name: 'Search entities' })).toHaveValue('');
    expect(screen.getByRole('menu', { name: 'Add entity' })).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole('searchbox', { name: 'Search entities' }), { key: 'Escape' });
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
