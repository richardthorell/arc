// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../editors/editorRegistry', () => ({ openAssetEditorDocument: vi.fn() }));

import { MaterialPicker } from './AssetPicker';

const readText = vi.fn();
const snapshot = vi.fn();
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

beforeEach(() => {
  readText.mockReset();
  snapshot.mockReset();
  snapshot.mockResolvedValue({
    activeProject: {
      projectRoot: 'D:/Project',
      descriptor: {
        paths: { content: 'Content' },
        assetRoots: ['Content'],
      },
    },
  });
  readText.mockResolvedValue({
    text: JSON.stringify({
      version: 4,
      name: 'Default Phong',
      graph: {
        version: 1,
        nodes: [
          {
            id: 'base-color',
            type: 'vector3',
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
          { id: 'material-output', type: 'output', position: [300, 0], values: {} },
        ],
        connections: [],
      },
    }),
  });
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: { projects: { readText, snapshot } },
  });
});

afterEach(cleanup);

describe('MaterialPicker exported parameters', () => {
  it('shows exported material defaults in a nested subsection', async () => {
    render(<MaterialPicker assets={[material]} label="Material" value={material.path} onChange={() => undefined} />);

    expect(screen.getByRole('region', { name: 'Material parameters' })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('2 exposed')).toBeVisible());

    expect(readText).toHaveBeenCalledWith(material.path, 'builtin');
    expect(screen.getByText('Base Color')).toBeVisible();
    expect(screen.getByLabelText('Base Color X')).toHaveTextContent('0.820');
    expect(screen.getByLabelText('Base Color Y')).toHaveTextContent('0.840');
    expect(screen.getByLabelText('Base Color Z')).toHaveTextContent('0.780');
    expect(screen.getByText('Roughness')).toBeVisible();
    expect(screen.getByLabelText('Roughness')).toHaveTextContent('0.620');
    expect(screen.queryByRole('spinbutton', { name: 'Roughness' })).not.toBeInTheDocument();
  });

  it('reads path-backed project parameters even before the asset list refreshes', async () => {
    const path = 'Content/Materials/New Material.arcmat';
    render(<MaterialPicker assets={[]} label="Material" value={path} onChange={() => undefined} />);

    expect(screen.getByRole('region', { name: 'Material parameters' })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('2 exposed')).toBeVisible());
    expect(readText).toHaveBeenCalledWith(path, 'project');
    expect(screen.getByText('Base Color')).toBeVisible();
    expect(screen.getByText('Roughness')).toBeVisible();
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

    await waitFor(() => expect(screen.getByText('2 exposed')).toBeVisible());
    expect(snapshot).toHaveBeenCalled();
    expect(readText).toHaveBeenCalledWith('Content/pr2.arcmat', 'project');
  });

  it('does not show shared material parameters for a mixed assignment', () => {
    render(
      <MaterialPicker assets={[material]} label="Material" mixed value={material.path} onChange={() => undefined} />,
    );

    expect(screen.queryByRole('region', { name: 'Material parameters' })).not.toBeInTheDocument();
    expect(readText).not.toHaveBeenCalled();
  });
});
