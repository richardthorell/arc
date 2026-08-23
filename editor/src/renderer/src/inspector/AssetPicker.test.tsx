// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AssetPicker, MaterialPicker } from './AssetPicker';

const writeText = vi.fn().mockResolvedValue({ succeeded: true });
const snapshot = vi.fn().mockResolvedValue({
  activeProject: {
    writable: true,
    projectRoot: 'D:/Test',
    descriptor: { paths: { content: 'GameContent' } },
  },
});

afterEach(cleanup);
beforeEach(() => {
  writeText.mockClear();
  snapshot.mockClear();
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: {
      projects: { snapshot, writeText },
    },
  });
});

describe('AssetPicker', () => {
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

    await waitFor(() => expect(snapshot).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(writeText.mock.calls[0][0]).toBe('GameContent/Hero Surface.arcmat');
    const asset = JSON.parse(writeText.mock.calls[0][1]);
    expect(asset.version).toBe(4);
    expect(asset).not.toHaveProperty('shader');
    expect(asset).not.toHaveProperty('surface');
    expect(asset).not.toHaveProperty('textures');
    expect(asset).not.toHaveProperty('advanced');
    expect(asset.graph.version).toBe(1);
    expect(asset.graph.nodes.some((node: { type: string }) => node.type === 'output')).toBe(true);
    await waitFor(() => expect(onChange).toHaveBeenCalledWith('GameContent/Hero Surface.arcmat'));
  });
});
