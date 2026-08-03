// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AssetPicker } from './AssetPicker';

afterEach(cleanup);

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
});
