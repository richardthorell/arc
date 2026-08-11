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

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver);
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
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('ViewportPanel', () => {
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

    await waitFor(() => expect(cameraInput).toHaveBeenCalledWith({ orbitX: 12, orbitY: -6 }));
    expect(cameraInput).not.toHaveBeenCalledWith(expect.objectContaining({ lookX: expect.any(Number) }));
  });
});
