// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { HostProjectComponentSchema } from './componentSchemas';
import { AddComponentPicker } from './AddComponentPicker';
import type { InspectorEntitySnapshot } from './inspectorTypes';

const snapshot = (overrides: Partial<InspectorEntitySnapshot> = {}): InspectorEntitySnapshot =>
  ({
    entity: { index: 1, generation: 1 },
    name: 'Entity',
    tag: 'Untagged',
    active: true,
    renderLayerMask: 1,
    transform: null,
    camera: null,
    light: null,
    meshRenderer: null,
    terrain: null,
    prefab: null,
    components: [],
    projectComponents: [],
    ...overrides,
  }) as InspectorEntitySnapshot;

const projectSchema = (overrides: Partial<HostProjectComponentSchema> = {}): HostProjectComponentSchema => ({
  id: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  name: 'gameplay_stats_component',
  displayName: 'Gameplay Stats',
  category: 'Gameplay',
  schemaVersion: 1,
  projectComponent: true,
  fields: [],
  ...overrides,
});

afterEach(cleanup);

describe('AddComponentPicker', () => {
  it('opens a searchable categorized picker and focuses search', async () => {
    render(<AddComponentPicker snapshot={snapshot()} projectSchemas={[]} onAdd={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Add Component' }));

    const search = screen.getByLabelText('Search add components');
    expect(search).toHaveFocus();
    expect(screen.getByText('Rendering')).toBeInTheDocument();
    expect(screen.getByText('Lighting')).toBeInTheDocument();

    await userEvent.type(search, 'spot');
    expect(screen.getByRole('button', { name: 'Spot Light' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Camera' })).not.toBeInTheDocument();
  });

  it('closes when clicking outside the picker', async () => {
    render(
      <div>
        <AddComponentPicker snapshot={snapshot()} projectSchemas={[]} onAdd={vi.fn()} />
        <button type="button">Outside</button>
      </div>,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Add Component' }));
    await userEvent.type(screen.getByLabelText('Search add components'), 'camera');

    await userEvent.click(screen.getByRole('button', { name: 'Outside' }));

    expect(screen.queryByRole('dialog', { name: 'Add Component' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Add Component' }));
    expect(screen.getByLabelText('Search add components')).toHaveValue('');
  });

  it('hides built-in single-instance components that are already attached', async () => {
    render(
      <AddComponentPicker
        snapshot={snapshot({
          camera: {} as InspectorEntitySnapshot['camera'],
          light: {} as InspectorEntitySnapshot['light'],
        })}
        projectSchemas={[]}
        onAdd={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Add Component' }));

    expect(screen.queryByRole('button', { name: 'Camera' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Directional Light' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Point Light' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Mesh Renderer' })).toBeInTheDocument();
  });

  it('honors allowMultiple for project component schemas', async () => {
    const attached = snapshot({
      projectComponents: [
        {
          typeId: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
          canonicalName: 'gameplay_stats_component',
          displayName: 'Gameplay Stats',
          schemaVersion: 1,
          values: {},
        },
      ],
    });
    render(
      <AddComponentPicker
        snapshot={attached}
        projectSchemas={[
          projectSchema(),
          projectSchema({
            id: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            name: 'audio_emitter_component',
            displayName: 'Audio Emitter',
            allowMultiple: true,
          }),
        ]}
        onAdd={vi.fn()}
      />,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Add Component' }));

    expect(screen.queryByRole('button', { name: 'Gameplay Stats' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Audio Emitter' })).toBeInTheDocument();
  });

  it('adds the selected component and closes after success', async () => {
    const onAdd = vi.fn().mockResolvedValue(true);
    render(<AddComponentPicker snapshot={snapshot()} projectSchemas={[]} onAdd={onAdd} />);
    await userEvent.click(screen.getByRole('button', { name: 'Add Component' }));
    await userEvent.click(screen.getByRole('button', { name: 'Mesh Renderer' }));

    expect(onAdd).toHaveBeenCalledWith('meshRenderer', 'Mesh Renderer');
    expect(screen.queryByRole('dialog', { name: 'Add Component' })).not.toBeInTheDocument();
  });
});
