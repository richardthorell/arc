// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../editors/editorRegistry', () => ({ openAssetEditorDocument: vi.fn() }));

import { MaterialPicker } from './AssetPicker';

const readText = vi.fn();
const snapshot = vi.fn();
const query = vi.fn();
const command = vi.fn();
const material = {
  id: 'default-phong',
  guid: 'default-phong-guid',
  name: 'default_phong.arcmat',
  path: 'assets/materials/default_phong.arcmat',
  kind: 'material',
  status: 'ready' as const,
  scope: 'builtin' as const,
  readOnly: true,
};
const texture = {
  id: 'albedo',
  guid: 'albedo-guid',
  name: 'albedo.png',
  path: 'assets/textures/albedo.png',
  kind: 'texture',
  status: 'ready' as const,
  scope: 'project' as const,
};

beforeEach(() => {
  readText.mockReset();
  snapshot.mockReset();
  query.mockReset();
  command.mockReset();
  snapshot.mockResolvedValue({
    activeProject: {
      projectRoot: 'D:/Project',
      descriptor: {
        paths: { content: 'Content' },
        assetRoots: ['Content'],
      },
    },
  });
  query.mockResolvedValue({
    succeeded: true,
    payload: {
      entity: { index: 7, generation: 2 },
      selectionCount: 1,
      meshRenderer: { materialName: 'Default Phong' },
    },
  });
  command.mockResolvedValue({ succeeded: true });
  readText.mockResolvedValue({
    text: JSON.stringify({
      version: 4,
      name: 'Default Phong',
      graph: {
        version: 1,
        nodes: [
          {
            id: 'base-color',
            type: 'colorRgb',
            position: [0, 0],
            values: { value: [0.82, 0.84, 0.78] },
            parameter: { exposed: true, name: 'Base Color' },
          },
          {
            id: 'roughness',
            type: 'constant',
            position: [0, 100],
            values: { value: 0.62 },
            parameter: { exposed: true, name: 'Roughness' },
          },
          {
            id: 'albedo',
            type: 'textureSample',
            position: [0, 200],
            values: { texture: 'assets/textures/default.png' },
            parameter: { exposed: true, name: 'Albedo' },
          },
          { id: 'material-output', type: 'output', position: [300, 0], values: {} },
        ],
        connections: [],
      },
    }),
  });
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: { projects: { readText, snapshot }, host: { query, command } },
  });
});

afterEach(cleanup);

describe('MaterialPicker exported parameters', () => {
  it('renders editable scalar, color, and texture instance controls', async () => {
    render(
      <MaterialPicker assets={[material, texture]} label="Material" value={material.path} onChange={() => undefined} />,
    );

    expect(screen.getByRole('region', { name: 'Material parameters' })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('3 exposed')).toBeVisible());

    expect(readText).toHaveBeenCalledWith(material.path, 'builtin');
    expect(screen.getByRole('button', { name: 'Open Base Color color picker' })).toBeVisible();
    expect(screen.getByLabelText('Roughness')).toHaveValue('0.620');
    expect(screen.getByRole('button', { name: 'Choose Albedo asset' })).toHaveTextContent('default');
  });

  it('uses the shared texture asset picker for texture overrides', async () => {
    render(
      <MaterialPicker assets={[material, texture]} label="Material" value={material.path} onChange={() => undefined} />,
    );

    const texturePicker = await screen.findByRole('button', { name: 'Choose Albedo asset' });
    fireEvent.click(texturePicker);
    fireEvent.click(await screen.findByRole('button', { name: 'Select albedo' }));

    await waitFor(() => expect(command).toHaveBeenCalledTimes(1));
    expect(command).toHaveBeenCalledWith(
      'entity.setMaterial',
      expect.objectContaining({
        path: expect.stringMatching(/^__arc_primitive_parameter__\/__arc_material_parameter__[0-9a-f]+\/0$/),
      }),
    );
  });

  it('creates a sparse instance override when a scalar changes', async () => {
    render(<MaterialPicker assets={[material]} label="Material" value={material.path} onChange={() => undefined} />);

    const roughness = await screen.findByLabelText('Roughness');
    fireEvent.change(roughness, { target: { value: '0.25' } });
    fireEvent.blur(roughness);

    await waitFor(() => expect(command).toHaveBeenCalledTimes(1));
    expect(command).toHaveBeenCalledWith(
      'entity.setMaterial',
      expect.objectContaining({
        entity: { index: 7, generation: 2 },
        applyToSelection: false,
        path: expect.stringMatching(/^__arc_primitive_parameter__\/__arc_material_parameter__[0-9a-f]+\/0$/),
      }),
    );
    expect(await screen.findByRole('button', { name: 'Reset Roughness' })).toBeVisible();
  });

  it('reads path-backed project parameters even before the asset list refreshes', async () => {
    const path = 'Content/Materials/New Material.arcmat';
    render(<MaterialPicker assets={[]} label="Material" value={path} onChange={() => undefined} />);

    expect(screen.getByRole('region', { name: 'Material parameters' })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('3 exposed')).toBeVisible());
    expect(readText).toHaveBeenCalledWith(path, 'project');
    expect(screen.getByLabelText('Roughness')).toBeVisible();
  });

  it('resolves a native asset-root-relative project material before reading parameters', async () => {
    const registryMaterial = {
      id: 'material-guid',
      guid: 'material-guid',
      name: 'pr2.arcmat',
      path: 'pr2.arcmat',
      kind: 'material',
      status: 'ready' as const,
      scope: 'project' as const,
    };
    render(
      <MaterialPicker
        assets={[registryMaterial]}
        label="Material"
        value={registryMaterial.path}
        onChange={() => undefined}
      />,
    );

    await waitFor(() => expect(screen.getByText('3 exposed')).toBeVisible());
    expect(snapshot).toHaveBeenCalled();
    expect(readText).toHaveBeenCalledWith('Content/pr2.arcmat', 'project');
  });

  it('does not show material parameters for a mixed assignment', () => {
    render(
      <MaterialPicker assets={[material]} label="Material" mixed value={material.path} onChange={() => undefined} />,
    );

    expect(screen.queryByRole('region', { name: 'Material parameters' })).not.toBeInTheDocument();
    expect(readText).not.toHaveBeenCalled();
  });
});
