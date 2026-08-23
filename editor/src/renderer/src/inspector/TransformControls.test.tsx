// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { inspectorComponentSchemas } from './componentSchemas';
import type { InspectorEntitySnapshot } from './inspectorTypes';
import { SchemaComponentCard } from './SchemaComponents';

const transformSchema = inspectorComponentSchemas.find((schema) => schema.id === 'transform')!;
const context: InspectorEntitySnapshot = {
  entity: { index: 1, generation: 1 },
  name: 'Transform Test Entity',
  tag: '',
  active: true,
  renderLayerMask: 1,
  mobility: 'movable',
  transform: {
    position: { x: 2, y: 3, z: 4 },
    rotationDegrees: { x: 10, y: 20, z: 30 },
    scale: { x: 2, y: 4, z: 6 },
    rotationQuaternion: { x: 0, y: 0, z: 0, w: 1 },
  },
  camera: null,
  light: null,
  meshRenderer: null,
  terrain: null,
  prefab: null,
  components: [],
  projectComponents: [],
};

afterEach(cleanup);

describe('Transform controls', () => {
  it('links scale axes by default and can unlink them', () => {
    const onValue = vi.fn();
    const view = render(
      <SchemaComponentCard
        schema={transformSchema}
        context={context}
        collapsed={false}
        onToggle={vi.fn()}
        onValue={onValue}
      />,
    );

    expect(view.getByRole('button', { name: 'Unlink scale axes' })).toBeInTheDocument();
    fireEvent.change(view.getByLabelText('Scale X'), { target: { value: '3' } });
    fireEvent.blur(view.getByLabelText('Scale X'));
    expect(onValue).toHaveBeenLastCalledWith('transform.scale', { x: 3, y: 6, z: 9 }, true);

    fireEvent.click(view.getByRole('button', { name: 'Unlink scale axes' }));
    fireEvent.change(view.getByLabelText('Scale X'), { target: { value: '5' } });
    fireEvent.blur(view.getByLabelText('Scale X'));
    expect(onValue).toHaveBeenLastCalledWith('transform.scale', { x: 5, y: 4, z: 6 }, true);
  });

  it('provides per-row reset actions and transform tooltips', () => {
    const onValue = vi.fn();
    const view = render(
      <SchemaComponentCard
        schema={transformSchema}
        context={context}
        collapsed={false}
        onToggle={vi.fn()}
        onValue={onValue}
      />,
    );

    fireEvent.click(view.getByRole('button', { name: 'Reset Location' }));
    expect(onValue).toHaveBeenCalledWith('transform.position', { x: 0, y: 0, z: 0 }, true);
    fireEvent.click(view.getByRole('button', { name: 'Reset Scale' }));
    expect(onValue).toHaveBeenCalledWith('transform.scale', { x: 1, y: 1, z: 1 }, true);
    expect(view.getAllByTitle(/Position relative to the parent/).length).toBeGreaterThan(0);
    expect(view.getAllByTitle(/Euler rotation in degrees/).length).toBeGreaterThan(0);
    expect(view.getAllByTitle(/Link axes to preserve proportions/).length).toBeGreaterThan(0);
  });
});
