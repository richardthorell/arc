// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { fireEvent, render, screen } from '@testing-library/react';
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
    const query = vi.fn().mockImplementation((type: string) =>
      Promise.resolve(
        type === 'texture.settings'
          ? {
              succeeded: true,
              payload: {
                settingsVersion: 4,
                preset: 'color',
                semantic: 'base_color',
                colorSpace: 'srgb',
                streamingMode: 'streamed_mips',
                compression: 'color',
                powerOfTwo: 'preserve',
                minFilter: 'linear',
                magFilter: 'linear',
                mipFilter: 'linear',
                wrapU: 'repeat',
                wrapV: 'repeat',
                mipGenerationFilter: 'box',
                maxSize: 8192,
                anisotropy: 8,
                lodBias: 0,
                minimumLod: 0,
                maximumLod: 1000,
                alphaCoverageThreshold: 0.5,
                generateMips: true,
                preserveAlphaCoverage: false,
              },
            }
          : {
              succeeded: true,
              payload: {
                path: textureDocument.path,
                width: 1024,
                height: 512,
                dataUrl: 'data:image/png;base64,preview',
              },
            },
      ),
    );
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: { host: { query } },
    });

    render(<TextureEditor document={textureDocument} />);

    expect(screen.getByRole('complementary', { name: 'Texture details' })).toBeInTheDocument();
    expect(screen.getByText('2048 × 1024')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Expand Streaming'));
    expect(screen.getByText('device')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Expand Import'));
    expect(screen.getByText('texture.image')).toBeInTheDocument();

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
