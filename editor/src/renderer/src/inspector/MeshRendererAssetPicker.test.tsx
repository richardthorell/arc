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
});
