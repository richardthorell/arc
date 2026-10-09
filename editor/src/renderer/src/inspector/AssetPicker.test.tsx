// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { arcAssetDragMime } from '../services/assetDragPayload';
import { AssetPicker, FlowPicker, MaterialPicker, TexturePicker } from './AssetPicker';

const createAsset = vi
  .fn()
  .mockImplementation(
    async ({ path, text: _text, kind }: { path: string; text: string; kind: 'material' | 'flow' | 'shader' }) => ({
      guid: `${kind}-asset-guid`,
      path,
      sourcePath: path,
      kind,
      state: 'ready',
    }),
  );
const snapshot = vi.fn().mockResolvedValue({
  activeProject: {
    writable: true,
    projectRoot: 'D:/Test',
    descriptor: { paths: { content: 'GameContent' } },
  },
});

afterEach(cleanup);
beforeEach(() => {
  createAsset.mockClear();
  snapshot.mockClear();
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: {
      projects: { snapshot, createAsset },
    },
  });
});

describe('AssetPicker', () => {
  it('accepts Content Browser material drag payloads on material slots', () => {
    const onChange = vi.fn();
    render(
      <MaterialPicker
        assets={[
          {
            id: 'hero-material-guid',
            guid: 'hero-material-guid',
            name: 'Hero Surface',
            path: 'Content/Materials/Hero.arcmat',
            kind: 'material',
            status: 'ready',
          },
        ]}
        label="Material"
        value=""
        onChange={onChange}
      />,
    );

    const control = screen.getByRole('button', { name: 'Choose Material asset' }).closest('.asset-reference-control');
    fireEvent.drop(control!, {
      dataTransfer: {
        dropEffect: 'none',
        getData: (type: string) =>
          type === arcAssetDragMime
            ? JSON.stringify({
                guid: 'hero-material-guid',
                type: 'material',
                pathHint: 'Content/Materials/Hero.arcmat',
              })
            : '',
      },
    });

    expect(onChange).toHaveBeenCalledWith('Content/Materials/Hero.arcmat');
  });

  it('accepts Content Browser image payloads on texture asset fields', () => {
    const onChange = vi.fn();
    render(
      <TexturePicker
        assets={[
          {
            id: 'albedo-guid',
            guid: 'albedo-guid',
            name: 'Albedo',
            path: 'Content/Textures/Albedo.png',
            kind: 'texture',
            status: 'ready',
          },
        ]}
        label="Albedo"
        referenceMode="guid"
        value=""
        onChange={onChange}
      />,
    );

    const control = screen.getByRole('button', { name: 'Choose Albedo asset' }).closest('.asset-reference-control');
    fireEvent.drop(control!, {
      dataTransfer: {
        dropEffect: 'none',
        getData: (type: string) =>
          type === arcAssetDragMime
            ? JSON.stringify({ guid: 'albedo-guid', type: 'texture', pathHint: 'Content/Textures/Albedo.png' })
            : '',
      },
    });

    expect(onChange).toHaveBeenCalledWith('albedo-guid');
  });

  it('uses the authoritative project source path when assigning imported textures', async () => {
    const onChange = vi.fn();
    const thumbnailProvider = vi.fn().mockResolvedValue('data:image/png;base64,AA');
    render(
      <TexturePicker
        assets={[
          {
            id: 'bricks-guid',
            guid: 'bricks-guid',
            name: 'Bricks071_4K_JPG_Color.jpg',
            path: 'Bricks071_4K_JPG_Color.jpg',
            sourcePath: 'Content/Bricks071_4K_JPG_Color.jpg',
            kind: 'texture',
            scope: 'project',
            status: 'ready',
          },
        ]}
        label="Base Color"
        thumbnailProvider={thumbnailProvider}
        value=""
        onChange={onChange}
      />,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Choose Base Color asset' }));
    await waitFor(() => expect(thumbnailProvider).toHaveBeenCalledWith('Bricks071_4K_JPG_Color.jpg'));
    await user.click(screen.getByRole('button', { name: 'Select Bricks071_4K_JPG_Color' }));

    expect(onChange).toHaveBeenCalledWith('Content/Bricks071_4K_JPG_Color.jpg');
  });

  it('filters reflected asset types and commits the stable GUID', async () => {
    const onChange = vi.fn();
    render(
      <AssetPicker
        assetKinds={['texture', 'mesh']}
        assetTypeIds={['texture-type']}
        assets={[
          {
            id: 'texture-guid',
            guid: 'texture-guid',
            typeId: 'texture-type',
            name: 'Albedo',
            path: 'Content/Albedo.png',
            kind: 'texture',
            status: 'ready',
          },
          {
            id: 'mesh-guid',
            guid: 'mesh-guid',
            typeId: 'mesh-type',
            name: 'Hero Mesh',
            path: 'Content/Hero.glb',
            kind: 'mesh',
            status: 'ready',
          },
        ]}
        label="Surface"
        referenceMode="guid"
        value=""
        onChange={onChange}
      />,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Choose Surface asset' }));
    expect(screen.getByRole('button', { name: 'Select Albedo' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Select Hero Mesh' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Select Albedo' }));
    expect(onChange).toHaveBeenCalledWith('texture-guid');
  });

  it('shows a friendly material identity and retries a thumbnail when importing becomes ready', async () => {
    const thumbnailProvider = vi
      .fn<(path: string) => Promise<string | null>>()
      .mockResolvedValueOnce(null)
      .mockResolvedValueOnce('data:image/bmp;base64,Qk');
    const importing = {
      id: 'material-guid',
      guid: 'material-guid',
      name: 'Antenna_Plastic.arcmat',
      path: 'Assets/imported/BistroExterior/materials/Antenna_Plastic.arcmat',
      kind: 'material',
      status: 'importing' as const,
      scope: 'project' as const,
    };
    const props = {
      assetKinds: ['material'],
      assetTypeLabel: 'Material',
      assets: [importing],
      label: 'Material',
      value: importing.path,
      thumbnailProvider,
      onChange: vi.fn(),
    };
    const { container, rerender } = render(<AssetPicker {...props} />);

    expect(screen.getByText('Antenna_Plastic')).toBeVisible();
    expect(screen.queryByText('Antenna_Plastic.arcmat')).not.toBeInTheDocument();
    expect(screen.getByText('Project Material')).toBeVisible();
    expect(screen.queryByText(importing.path)).not.toBeInTheDocument();
    await waitFor(() => expect(thumbnailProvider).toHaveBeenCalledTimes(1));

    rerender(<AssetPicker {...props} assets={[{ ...importing, status: 'ready' as const }]} />);
    await waitFor(() => expect(thumbnailProvider).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(container.querySelector('.asset-reference-main img')).not.toBeNull());
  });

  it('uses asset metadata titles for built-in material display names', async () => {
    render(
      <MaterialPicker
        assets={[
          {
            id: 'standard-lit-guid',
            guid: 'standard-lit-guid',
            name: 'default_phong.arcmat',
            title: 'Standard Lit',
            path: 'materials/default_phong.arcmat',
            kind: 'material',
            status: 'ready',
            scope: 'builtin',
            readOnly: true,
          },
        ]}
        label="Material"
        value="materials/default_phong.arcmat"
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByText('Standard Lit')).toBeVisible();
    expect(screen.queryByText('default_phong')).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Choose Material asset' }));
    expect(screen.getByRole('button', { name: 'Select Standard Lit' })).toBeVisible();
  });

  it('creates and assigns a new material from the real project content root', async () => {
    const onChange = vi.fn();
    render(
      <MaterialPicker
        assets={[
          {
            id: 'existing-material',
            name: 'Existing.arcmat',
            path: 'Content/Existing.arcmat',
            kind: 'material',
            status: 'ready',
            scope: 'project',
          },
        ]}
        label="Material"
        value="Content/Existing.arcmat"
        onChange={onChange}
      />,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Choose Material asset' }));
    await user.click(screen.getByRole('button', { name: 'Create New Material…' }));

    const dialog = screen.getByRole('dialog', { name: 'Material asset picker' });
    expect(dialog).toHaveStyle({ gridTemplateRows: '31px minmax(0, 1fr)' });
    expect(screen.getByText('Create Material')).toBeVisible();

    await user.clear(screen.getByLabelText('New material name'));
    await user.type(screen.getByLabelText('New material name'), 'Hero Surface');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(snapshot).toHaveBeenCalled());
    await waitFor(() => expect(createAsset).toHaveBeenCalledTimes(1));
    expect(createAsset.mock.calls[0][0].path).toBe('GameContent/Hero Surface.arcmat');
    const asset = JSON.parse(createAsset.mock.calls[0][0].text);
    expect(asset.version).toBe(4);
    expect(asset).not.toHaveProperty('shader');
    expect(asset).not.toHaveProperty('surface');
    expect(asset).not.toHaveProperty('textures');
    expect(asset).not.toHaveProperty('advanced');
    expect(asset.graph.version).toBe(1);
    expect(asset.graph.nodes.some((node: { type: string }) => node.type === 'output')).toBe(true);
    await waitFor(() => expect(onChange).toHaveBeenCalledWith('GameContent/Hero Surface.arcmat'));
  });
  it('creates and assigns a new Flow Graph from an empty asset picker', async () => {
    const onChange = vi.fn();
    render(<FlowPicker allowedExtensions={['.arcflow']} assets={[]} label="Graph" value="" onChange={onChange} />);

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Choose Graph asset' }));

    expect(screen.getByText('Select Flow Graph')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Create New Flow Graph…' })).toBeVisible();

    await user.click(screen.getByRole('button', { name: 'Create New Flow Graph…' }));
    await user.clear(screen.getByLabelText('New flow graph name'));
    await user.type(screen.getByLabelText('New flow graph name'), 'Game Startup');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(createAsset).toHaveBeenCalledTimes(1));
    expect(createAsset.mock.calls[0][0].path).toBe('GameContent/Game Startup.arcflow');
    const asset = JSON.parse(createAsset.mock.calls[0][0].text);
    expect(asset.assetType).toBe('flow');
    expect(asset.name).toBe('Game Startup');
    expect(asset.graph.version).toBe(1);
    await waitFor(() => expect(onChange).toHaveBeenCalledWith('flow-asset-guid'));
  });

  it('shows an open-in-editor action for an assigned Flow Graph', () => {
    render(
      <FlowPicker
        allowedExtensions={['.arcflow']}
        assets={[
          {
            id: 'game-startup-flow',
            name: 'Game Startup',
            path: 'GameContent/Game Startup.arcflow',
            kind: 'flow',
            status: 'ready',
            scope: 'project',
          },
        ]}
        label="Graph"
        value="GameContent/Game Startup.arcflow"
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('button', { name: 'Open Game Startup in Flow Graph Editor' })).toBeVisible();
  });

  it('rejects cubemaps in 2D Texture fields and accepts them in Cube Texture fields', async () => {
    const assets = [
      {
        id: 'albedo-guid',
        guid: 'albedo-guid',
        name: 'Albedo',
        path: 'Content/Textures/Albedo.png',
        kind: 'texture',
        textureDimension: '2d' as const,
        status: 'ready' as const,
      },
      {
        id: 'sky-guid',
        guid: 'sky-guid',
        name: 'Studio Sky',
        path: 'Content/Environments/Studio.hdr',
        kind: 'environment',
        textureDimension: 'cube' as const,
        status: 'ready' as const,
      },
    ];
    const on2DChange = vi.fn();
    const { unmount } = render(<TexturePicker assets={assets} label="Base Color" value="" onChange={on2DChange} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Choose Base Color asset' }));
    expect(screen.getByRole('button', { name: 'Select Studio Sky' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Select Studio Sky' })).toHaveAttribute(
      'title',
      'Expected 2D Texture · Cube Texture provided',
    );
    unmount();

    const onCubeChange = vi.fn();
    render(
      <TexturePicker
        assets={assets}
        expectedTextureDimension="cube"
        label="Environment"
        value=""
        onChange={onCubeChange}
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Choose Environment asset' }));
    expect(screen.getByRole('button', { name: 'Select Albedo' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Select Studio Sky' })).toBeEnabled();
  });
});
