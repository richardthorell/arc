// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ViewportPanel } from './ViewportPanel';

class TestResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}

let nextAnimationFrame: FrameRequestCallback | null = null;

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver);
  vi.stubGlobal(
    'requestAnimationFrame',
    vi.fn((callback: FrameRequestCallback) => {
      nextAnimationFrame = callback;
      return 1;
    }),
  );
  vi.stubGlobal('cancelAnimationFrame', vi.fn());
  Object.defineProperty(HTMLElement.prototype, 'setPointerCapture', {
    configurable: true,
    value: vi.fn(),
  });
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue({
    x: 0,
    y: 0,
    top: 0,
    left: 0,
    right: 640,
    bottom: 480,
    width: 640,
    height: 480,
    toJSON: () => ({}),
  });
});

afterEach(() => {
  nextAnimationFrame = null;
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('ViewportPanel', () => {
  it('uses a DOM canvas and shared viewport transport when streaming is available', async () => {
    const create = vi.fn().mockResolvedValue({ succeeded: true });
    const registerSurface = vi.fn();
    const pointer = vi.fn().mockResolvedValue({ succeeded: true });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: {
          query: vi.fn().mockResolvedValue({
            succeeded: true,
            payload: {
              width: 1280,
              height: 960,
              fps: 60,
              frameTimeMs: 1,
              drawCalls: 1,
              frameIndex: 2,
              submitted: true,
            },
          }),
        },
        viewport: {
          create,
          attach: vi.fn(),
          resize: vi.fn().mockResolvedValue({ succeeded: true }),
          detach: vi.fn().mockResolvedValue({ succeeded: true }),
          cameraInput: vi.fn().mockResolvedValue({ succeeded: true }),
          pointer,
          key: vi.fn(),
          registerSurface,
          unregisterSurface: vi.fn(),
          setVisibility: vi.fn(),
        },
      },
    });

    const view = render(
      <ViewportPanel
        project={null}
        startupState={{ appVersion: '0.1.0', engineHostConnected: true, viewportMode: 'streamed' }}
        onCommand={vi.fn()}
        onReconnect={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    await waitFor(() => expect(create).toHaveBeenCalled());
    expect(registerSurface).toHaveBeenCalledWith('viewport-1', expect.stringContaining('viewport-1'));
    expect(view.getByText('GPU Shared')).toBeInTheDocument();
    const canvas = view.getByLabelText('ARC 3D viewport');
    expect(canvas).toBeInstanceOf(HTMLCanvasElement);
    fireEvent.pointerMove(canvas.parentElement!, { clientX: 320, clientY: 240, pointerId: 1 });
    await waitFor(() => expect(pointer).toHaveBeenCalledWith(expect.objectContaining({ phase: 'move' })));
  });

  it('requires explicit confirmation before falling back from shared GPU rendering', async () => {
    const create = vi.fn().mockResolvedValue({ succeeded: false, error: 'D3D12 shared handle import failed' });
    const attach = vi.fn().mockResolvedValue({ succeeded: true });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: {
          query: vi.fn().mockResolvedValue({
            succeeded: true,
            payload: {
              width: 640,
              height: 480,
              fps: 0,
              frameTimeMs: 0,
              drawCalls: 0,
              frameIndex: 0,
              submitted: false,
            },
          }),
        },
        viewport: {
          create,
          attach,
          resize: vi.fn().mockResolvedValue({ succeeded: true }),
          detach: vi.fn().mockResolvedValue({ succeeded: true }),
          cameraInput: vi.fn().mockResolvedValue({ succeeded: true }),
          pointer: vi.fn().mockResolvedValue({ succeeded: true }),
          key: vi.fn().mockResolvedValue({ succeeded: true }),
          registerSurface: vi.fn(),
          unregisterSurface: vi.fn(),
          setVisibility: vi.fn(),
        },
      },
    });

    const view = render(
      <ViewportPanel
        project={null}
        startupState={{ appVersion: '0.1.0', engineHostConnected: true, viewportMode: 'streamed' }}
        onCommand={vi.fn()}
        onReconnect={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    await waitFor(() => expect(view.getByText('Shared GPU viewport failed')).toBeInTheDocument());
    expect(view.getByText('D3D12 shared handle import failed')).toBeInTheDocument();
    expect(attach).not.toHaveBeenCalled();
    expect(view.getByText('GPU Shared')).toHaveAttribute(
      'title',
      expect.stringContaining('D3D12 shared handle import failed'),
    );

    fireEvent.click(view.getByRole('button', { name: 'Use Native Fallback' }));

    await waitFor(() => expect(attach).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(view.getByText('Native Fallback')).toBeInTheDocument());
    expect(view.getByText('Native Fallback')).toHaveAttribute(
      'title',
      expect.stringContaining('D3D12 shared handle import failed'),
    );
  });

  it('resizes the native viewport when docking changes its position without changing its size', async () => {
    let left = 40;
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => ({
      x: left,
      y: 80,
      top: 80,
      left,
      right: left + 640,
      bottom: 560,
      width: 640,
      height: 480,
      toJSON: () => ({}),
    }));
    const resize = vi.fn().mockResolvedValue({ succeeded: true });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: {
          query: vi.fn().mockResolvedValue({
            succeeded: true,
            payload: {
              width: 640,
              height: 480,
              fps: 60,
              frameTimeMs: 16.6,
              drawCalls: 1,
              frameIndex: 1,
              submitted: true,
            },
          }),
        },
        viewport: {
          attach: vi.fn().mockResolvedValue({ succeeded: true }),
          resize,
          detach: vi.fn().mockResolvedValue({ succeeded: true }),
          cameraInput: vi.fn().mockResolvedValue({ succeeded: true }),
        },
      },
    });

    render(
      <ViewportPanel
        project={null}
        startupState={{ appVersion: '0.1.0', engineHostConnected: true, viewportMode: 'native' }}
        onCommand={vi.fn()}
        onReconnect={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    await waitFor(() =>
      expect(window.arc.viewport.attach).toHaveBeenCalledWith(
        expect.objectContaining({ x: 40, y: 80, width: 640, height: 480 }),
      ),
    );
    left = 260;
    nextAnimationFrame?.(16);

    await waitFor(() =>
      expect(resize).toHaveBeenCalledWith(expect.objectContaining({ x: 260, y: 80, width: 640, height: 480 })),
    );
  });

  it('toggles the adaptive grid through the viewport render options', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: {
          command,
          query: vi.fn().mockResolvedValue({
            succeeded: true,
            payload: {
              width: 640,
              height: 480,
              fps: 60,
              frameTimeMs: 16.6,
              drawCalls: 1,
              frameIndex: 1,
              submitted: true,
              renderOptions: {
                renderMode: 'shaded',
                visualization: 'none',
                shadows: true,
                environment: true,
                lighting: true,
                grid: true,
              },
            },
          }),
        },
        viewport: {
          attach: vi.fn().mockResolvedValue({ succeeded: true }),
          resize: vi.fn().mockResolvedValue({ succeeded: true }),
          detach: vi.fn().mockResolvedValue({ succeeded: true }),
          cameraInput: vi.fn().mockResolvedValue({ succeeded: true }),
        },
      },
    });

    const view = render(
      <ViewportPanel
        project={null}
        startupState={{ appVersion: '0.1.0', engineHostConnected: true, viewportMode: 'native' }}
        onCommand={vi.fn()}
        onReconnect={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    fireEvent.click(view.getByText('Show'));
    const grid = await view.findByRole('menuitemcheckbox', { name: /Grid/ });
    expect(grid).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(grid);

    await waitFor(() =>
      expect(command).toHaveBeenCalledWith(
        'viewport.setRenderOptions',
        expect.objectContaining({ grid: false, renderMode: 'shaded' }),
      ),
    );
  });

  it('selects texture residency debug visualization modes', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: {
          command,
          query: vi.fn().mockResolvedValue({
            succeeded: true,
            payload: {
              width: 640,
              height: 480,
              fps: 60,
              frameTimeMs: 16.6,
              drawCalls: 1,
              frameIndex: 1,
              submitted: true,
              renderOptions: {
                renderMode: 'shaded',
                visualization: 'standard',
                shadows: true,
                environment: true,
                lighting: true,
                grid: true,
              },
            },
          }),
        },
        viewport: {
          attach: vi.fn().mockResolvedValue({ succeeded: true }),
          resize: vi.fn().mockResolvedValue({ succeeded: true }),
          detach: vi.fn().mockResolvedValue({ succeeded: true }),
          cameraInput: vi.fn().mockResolvedValue({ succeeded: true }),
        },
      },
    });

    const view = render(
      <ViewportPanel
        project={null}
        startupState={{ appVersion: '0.1.0', engineHostConnected: true, viewportMode: 'native' }}
        onCommand={vi.fn()}
        onReconnect={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    fireEvent.click(view.getAllByText('Lit')[0]);
    fireEvent.click(await view.findByText('Texture Desired Mip'));

    await waitFor(() =>
      expect(command).toHaveBeenCalledWith(
        'viewport.setRenderOptions',
        expect.objectContaining({ renderMode: 'shaded', visualization: 'textureDesiredMip' }),
      ),
    );
  });

  it('retries an idle native viewport attachment until rendering starts', async () => {
    const attach = vi.fn().mockResolvedValue({ succeeded: true });
    const resize = vi.fn().mockResolvedValue({ succeeded: true });
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      payload: {
        width: 640,
        height: 480,
        fps: 0,
        frameTimeMs: 0,
        drawCalls: 0,
        frameIndex: 0,
        submitted: false,
      },
    });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: { query },
        viewport: {
          attach,
          resize,
          detach: vi.fn().mockResolvedValue({ succeeded: true }),
          cameraInput: vi.fn().mockResolvedValue({ succeeded: true }),
        },
      },
    });

    render(
      <ViewportPanel
        project={null}
        startupState={{ appVersion: '0.1.0', engineHostConnected: true, viewportMode: 'native' }}
        onCommand={vi.fn()}
        onReconnect={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    await waitFor(() => expect(attach).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(attach.mock.calls.length).toBeGreaterThanOrEqual(2), { timeout: 2200 });
  });

  it('latches Alt + left-drag as focused orbit for the complete gesture', async () => {
    const cameraInput = vi.fn().mockResolvedValue({ succeeded: true });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: {
          query: vi.fn().mockResolvedValue({
            succeeded: true,
            payload: {
              width: 640,
              height: 480,
              fps: 60,
              frameTimeMs: 16.6,
              drawCalls: 1,
              frameIndex: 1,
              submitted: true,
            },
          }),
        },
        viewport: {
          attach: vi.fn().mockResolvedValue({ succeeded: true }),
          resize: vi.fn().mockResolvedValue({ succeeded: true }),
          detach: vi.fn().mockResolvedValue({ succeeded: true }),
          cameraInput,
        },
      },
    });

    const view = render(
      <ViewportPanel
        project={null}
        startupState={{ appVersion: '0.1.0', engineHostConnected: true, viewportMode: 'native' }}
        onCommand={vi.fn()}
        onReconnect={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    const viewport = view.container.querySelector('.arc-viewport-body');
    expect(viewport).not.toBeNull();

    fireEvent.pointerDown(viewport!, { pointerId: 7, button: 0, clientX: 20, clientY: 30, altKey: true });
    fireEvent.pointerMove(viewport!, { pointerId: 7, clientX: 32, clientY: 24, altKey: false });

    await waitFor(() => expect(cameraInput).toHaveBeenCalledWith({ viewportId: 'viewport-1', orbitX: 12, orbitY: -6 }));
    expect(cameraInput).not.toHaveBeenCalledWith(expect.objectContaining({ lookX: expect.any(Number) }));
  });
});
