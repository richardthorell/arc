// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { TextureEditor } from './TextureEditor';
import { getTextureEditorViewState, setTextureEditorViewState } from './textureEditorViewState';

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
  cleanup();
  vi.restoreAllMocks();
  Reflect.deleteProperty(window, 'arc');
  setTextureEditorViewState(textureDocument.id, { zoom: 1, previewMode: 'processed' });
});

describe('TextureEditor', () => {
  it('uses the native surface for the default processed view without requesting a thumbnail', async () => {
    const query = vi.fn().mockResolvedValue({ succeeded: false });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        getStartupState: vi.fn().mockResolvedValue({ engineHostConnected: true, viewportMode: 'streamed' }),
        host: { query },
        viewport: { registerSurface: vi.fn(), unregisterSurface: vi.fn() },
      },
    });

    render(<TextureEditor document={textureDocument} />);

    expect(await screen.findByRole('img', { name: 'T_Rock.png texture preview' })).toContainElement(
      document.querySelector('.asset-preview-viewport-canvas'),
    );
    expect(query).not.toHaveBeenCalledWith('asset.thumbnail', expect.anything());
  });

  it('keeps processed rulers and scroll canvas while zooming locally', async () => {
    const cameraInput = vi.fn();
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        getStartupState: vi.fn().mockResolvedValue({ engineHostConnected: true, viewportMode: 'streamed' }),
        host: { query: vi.fn().mockResolvedValue({ succeeded: false }) },
        viewport: { cameraInput, registerSurface: vi.fn(), unregisterSurface: vi.fn() },
      },
    });
    const { container } = render(<TextureEditor document={textureDocument} />);
    const native = await screen.findByRole('img', { name: 'T_Rock.png texture preview' });

    expect(container.querySelector('.texture-ruler-horizontal')).toBeInTheDocument();
    expect(container.querySelector('.texture-ruler-vertical')).toBeInTheDocument();
    expect(native.closest('.texture-preview-canvas')).toBeInTheDocument();
    expect(native.closest('.texture-preview-scroll')).toBeInTheDocument();

    fireEvent.wheel(native, { deltaY: -100 });
    expect(getTextureEditorViewState(textureDocument.id).zoom).toBeCloseTo(1.12);
    expect(container.querySelector('.texture-native-surface')).toHaveStyle({ transform: 'scale(1.12)' });

    const stage = container.querySelector('.texture-preview-stage') as HTMLElement;
    const scroll = container.querySelector('.texture-preview-scroll') as HTMLElement;
    stage.setPointerCapture = vi.fn();
    fireEvent.pointerDown(native, { button: 1, pointerId: 7, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(stage, { pointerId: 7, clientX: 50, clientY: 80 });
    expect(scroll.scrollLeft).toBe(50);
    expect(scroll.scrollTop).toBe(20);
    expect(cameraInput).not.toHaveBeenCalled();
  });

  it('fits the processed texture to the available viewport on first open', async () => {
    vi.spyOn(Element.prototype, 'clientWidth', 'get').mockReturnValue(1000);
    vi.spyOn(Element.prototype, 'clientHeight', 'get').mockReturnValue(800);
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        getStartupState: vi.fn().mockResolvedValue({ engineHostConnected: false, viewportMode: 'native' }),
        host: { query: vi.fn().mockResolvedValue({ succeeded: false }) },
      },
    });

    render(<TextureEditor document={textureDocument} />);

    await waitFor(() => expect(getTextureEditorViewState(textureDocument.id).zoom).toBeCloseTo(944 / 2048));
  });

  it('renders texture metadata and falls back cleanly when the native preview is unavailable', async () => {
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
                mipPolicy: 'preserve_source',
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
          : { succeeded: false, error: `Unexpected query: ${type}` },
      ),
    );
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        getStartupState: vi.fn().mockResolvedValue({ engineHostConnected: false, viewportMode: 'native' }),
        host: { query },
      },
    });

    render(<TextureEditor document={textureDocument} />);

    expect(screen.getByRole('complementary', { name: 'Texture details' })).toBeInTheDocument();
    expect(screen.getByLabelText('Texture navigation controls')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Fit texture to screen' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Zoom out' })).toBeInTheDocument();
    expect(screen.getByLabelText('Texture zoom')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reset zoom to 100%' })).toBeInTheDocument();
    expect(screen.getByText('2048 × 1024')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Expand Streaming'));
    expect(screen.getByText('device')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Expand Import'));
    expect(screen.getByText('texture.image')).toBeInTheDocument();

    expect(await screen.findByText('GPU preview unavailable')).toBeInTheDocument();
    expect(query).toHaveBeenCalledWith('texture.settings', { guid: 'texture-guid' });
    expect(query).not.toHaveBeenCalledWith('asset.thumbnail', expect.anything());
  });
  it('shows DDS authored mip policy and reports compressed preview limitations', async () => {
    const ddsDocument: EditorDocument = {
      ...textureDocument,
      id: 'texture:dds-guid',
      title: 'T_Authored.dds',
      path: 'Content/Textures/T_Authored.dds',
      assetId: 'dds-guid',
      assetGuid: 'dds-guid',
      assetSnapshot: {
        ...textureDocument.assetSnapshot!,
        id: 'dds-guid',
        guid: 'dds-guid',
        name: 'T_Authored.dds',
        path: 'Content/Textures/T_Authored.dds',
        textureFormat: 'BC7 RGBA sRGB',
        mipLevels: 13,
      },
    };
    const query = vi.fn().mockImplementation((type: string) =>
      Promise.resolve(
        type === 'texture.settings'
          ? {
              succeeded: true,
              payload: {
                settingsVersion: 8,
                preset: 'color',
                semantic: 'base_color',
                colorSpace: 'srgb',
                streamingMode: 'streamed_mips',
                compression: 'color',
                powerOfTwo: 'preserve',
                minFilter: 'linear',
                magFilter: 'linear',
                mipFilter: 'linear',
                mipPolicy: 'preserve_source',
                wrapU: 'repeat',
                wrapV: 'repeat',
                mipGenerationFilter: 'kaiser',
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
          : { succeeded: false, error: 'Texture thumbnail could not be generated' },
      ),
    );
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        getStartupState: vi.fn().mockResolvedValue({ engineHostConnected: false, viewportMode: 'native' }),
        host: { query },
      },
    });

    render(<TextureEditor document={ddsDocument} />);

    expect(await screen.findByText('Authored / preserved')).toBeInTheDocument();
    expect(screen.getAllByText('13')).toHaveLength(2);
    const mipPolicy = screen.getByRole('combobox', { name: 'Texture mip policy' });
    expect(mipPolicy).toHaveTextContent('Preserve Source');
    fireEvent.click(mipPolicy);
    expect(screen.getByRole('option', { name: 'Generate' })).toBeDisabled();
    expect(await screen.findByText('GPU preview unavailable')).toBeInTheDocument();
    expect(query).not.toHaveBeenCalledWith('asset.thumbnail', expect.anything());
  });
});
