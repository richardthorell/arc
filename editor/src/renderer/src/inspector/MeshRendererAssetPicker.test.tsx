// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SchemaComponentCard } from './SchemaComponents';

afterEach(cleanup);

describe('Mesh Renderer asset picker', () => {
  it('shows the authored mesh and routes a new mesh selection through the host assignment command path', async () => {
    const onValue = vi.fn();
    render(
      <SchemaComponentCard
        assets={[
          {
            id: 'mesh-cabin',
            name: 'SM_Cabin',
            path: 'Assets/Environment/Cabins/SM_Cabin.glb',
            kind: 'mesh',
            status: 'ready',
          },
          {
            id: 'mesh-rock',
            name: 'SM_Rock',
            path: 'Assets/Environment/Rocks/SM_Rock.glb',
            kind: 'scene',
            status: 'ready',
          },
        ]}
        collapsed={false}
        context={{ meshRenderer: { meshPath: 'Assets/Environment/Cabins/SM_Cabin.glb' } }}
        schema={{ id: 'meshRenderer', title: 'Mesh Renderer', fields: [] }}
        onToggle={() => undefined}
        onValue={onValue}
      />,
    );

    expect(screen.getByLabelText('Choose Mesh asset')).toBeInTheDocument();
    expect(screen.getByText('SM_Cabin')).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText('Choose Mesh asset'));
    await userEvent.click(screen.getByLabelText('Select SM_Rock'));

    expect(onValue).toHaveBeenCalledWith(
      'meshRenderer.materialPath',
      '__arc_mesh__/Assets/Environment/Rocks/SM_Rock.glb',
      true,
    );
  });

  it('shows built-in primitives as procedural meshes with shape icons and routes selections to the host', async () => {
    const onValue = vi.fn();
    render(
      <SchemaComponentCard
        collapsed={false}
        context={{ meshRenderer: { meshPath: 'arc://primitive/sphere' } }}
        schema={{ id: 'meshRenderer', title: 'Mesh Renderer', fields: [] }}
        onToggle={() => undefined}
        onValue={onValue}
      />,
    );

    expect(screen.getByText('Sphere')).toBeInTheDocument();
    expect(screen.getByText('Procedural Mesh')).toBeInTheDocument();
    expect(screen.getByTestId('primitive-mesh-icon-sphere')).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText('Choose Mesh asset'));
    expect(screen.getByLabelText('Select Plane')).toBeInTheDocument();
    expect(screen.getByLabelText('Select Cube')).toBeInTheDocument();
    expect(screen.getByLabelText('Select Sphere')).toBeInTheDocument();
    expect(screen.getByLabelText('Select Cylinder')).toBeInTheDocument();
    expect(screen.getByLabelText('Select Cone')).toBeInTheDocument();
    expect(screen.getByLabelText('Select Capsule')).toBeInTheDocument();
    expect(screen.getByTestId('primitive-mesh-icon-plane')).toBeInTheDocument();
    expect(screen.getByTestId('primitive-mesh-icon-cube')).toBeInTheDocument();
    expect(screen.getAllByTestId('primitive-mesh-icon-sphere')).toHaveLength(2);
    expect(screen.getByTestId('primitive-mesh-icon-cylinder')).toBeInTheDocument();
    expect(screen.getByTestId('primitive-mesh-icon-cone')).toBeInTheDocument();
    expect(screen.getByTestId('primitive-mesh-icon-capsule')).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText('Select Cube'));

    expect(onValue).toHaveBeenCalledWith('meshRenderer.materialPath', '__arc_primitive__/cube', true);
  });
});
