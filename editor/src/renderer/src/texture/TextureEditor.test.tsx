// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { TextureEditor } from './TextureEditor';

const textureDocument: EditorDocument = {
  id: 'texture:texture-guid',
  kind: 'texture',
  title: 'T_Rock.png',
  path: 'Content/Textures/T_Rock.png',
  assetId: 'texture-guid',
  assetGuid: 'texture-guid',
  assetScope: 'project',
  assetSnapshot: {
    id: 'texture-guid',
    guid: 'texture-guid',
    name: 'T_Rock.png',
    path: 'Content/Textures/T_Rock.png',
    scope: 'project',
    kind: 'texture',
    status: 'ready',
    readOnly: false,
    residency: 'device',
    importerId: 'texture.image',
    sourceBytes: 1_572_864,
    width: 2048,
    height: 1024,
    mipLevels: 12,
  },
  dirty: false,
  readOnly: false,
};

afterEach(() => {
  vi.restoreAllMocks();
  Reflect.deleteProperty(window, 'arc');
});

describe('TextureEditor', () => {
  it('renders texture metadata and requests a large preview from the shared asset thumbnail host', async () => {
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      payload: {
        path: textureDocument.path,
        width: 1024,
        height: 512,
        dataUrl: 'data:image/png;base64,preview',
      },
    });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: { host: { query } },
    });

    render(<TextureEditor document={textureDocument} />);

    expect(screen.getByRole('complementary', { name: 'Texture details' })).toBeInTheDocument();
    expect(screen.getByText('2048 × 1024')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('texture.image')).toBeInTheDocument();
    expect(screen.getByText('device')).toBeInTheDocument();
    expect(await screen.findByAltText('T_Rock.png texture preview')).toHaveAttribute(
      'src',
      'data:image/png;base64,preview',
    );
    expect(query).toHaveBeenCalledWith('asset.thumbnail', {
      path: 'Content/Textures/T_Rock.png',
      maxSize: 2048,
    });
  });
});
