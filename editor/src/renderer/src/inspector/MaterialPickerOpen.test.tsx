// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

const { openAssetEditorDocument } = vi.hoisted(() => ({
  openAssetEditorDocument: vi.fn(),
}));

vi.mock('../editors/editorRegistry', () => ({ openAssetEditorDocument }));

import { MaterialPicker } from './AssetPicker';

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('MaterialPicker editor action', () => {
  it('opens the assigned material in the material editor without opening the picker', async () => {
    const material = {
      id: 'material-guid',
      guid: 'material-guid',
      typeId: 'material-type',
      name: 'New Material.arcmat',
      path: 'Content/Materials/New Material.arcmat',
      kind: 'material',
      status: 'ready' as const,
      scope: 'project' as const,
    };

    render(
      <MaterialPicker
        assets={[material]}
        label="Material"
        value={material.path}
        onChange={() => undefined}
      />,
    );

    const openButton = screen.getByRole('button', { name: 'Open New Material in Material Editor' });
    expect(openButton).toHaveAttribute('title', 'Open in Material Editor');

    await userEvent.click(openButton);

    expect(openAssetEditorDocument).toHaveBeenCalledTimes(1);
    expect(openAssetEditorDocument).toHaveBeenCalledWith(
      expect.objectContaining({
        id: material.id,
        guid: material.guid,
        typeId: material.typeId,
        name: material.name,
        path: material.path,
        kind: 'material',
        status: 'ready',
        scope: 'project',
      }),
    );
    expect(screen.queryByRole('dialog', { name: 'Material asset picker' })).not.toBeInTheDocument();
  });

  it('does not offer the editor action for an unassigned or mixed material value', () => {
    const material = {
      id: 'material-guid',
      name: 'Surface.arcmat',
      path: 'Content/Materials/Surface.arcmat',
      kind: 'material',
      status: 'ready' as const,
      scope: 'project' as const,
    };

    const { rerender } = render(
      <MaterialPicker assets={[material]} label="Material" value="" onChange={() => undefined} />,
    );
    expect(screen.queryByRole('button', { name: /in Material Editor$/ })).not.toBeInTheDocument();

    rerender(
      <MaterialPicker assets={[material]} label="Material" mixed value={material.path} onChange={() => undefined} />,
    );
    expect(screen.queryByRole('button', { name: /in Material Editor$/ })).not.toBeInTheDocument();
  });
});
