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

    await waitFor(() => expect(window.arc.viewport.attach).toHaveBeenCalledWith(
      expect.objectContaining({ x: 40, y: 80, width: 640, height: 480 }),
    ));
    left = 260;
    nextAnimationFrame?.(16);

    await waitFor(() => expect(resize).toHaveBeenCalledWith(
      expect.objectContaining({ x: 260, y: 80, width: 640, height: 480 }),
    ));
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

    await waitFor(() =>
      expect(cameraInput).toHaveBeenCalledWith({ viewportId: 'viewport-1', orbitX: 12, orbitY: -6 }),
    );
    expect(cameraInput).not.toHaveBeenCalledWith(expect.objectContaining({ lookX: expect.any(Number) }));
  });
});
