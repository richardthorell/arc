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
  vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1));
  vi.stubGlobal('cancelAnimationFrame', vi.fn());
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

describe('ViewportPanel wheel input', () => {
  it('routes both wheel directions through signed camera input in streamed mode', async () => {
    const cameraInput = vi.fn().mockResolvedValue({ succeeded: true });
    const pointer = vi.fn().mockResolvedValue({ succeeded: true });
    const create = vi.fn().mockResolvedValue({ succeeded: true });

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
          create,
          attach: vi.fn(),
          resize: vi.fn().mockResolvedValue({ succeeded: true }),
          detach: vi.fn().mockResolvedValue({ succeeded: true }),
          cameraInput,
          pointer,
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

    await waitFor(() => expect(create).toHaveBeenCalled());
    const viewport = view.container.querySelector('.arc-viewport-body');
    expect(viewport).not.toBeNull();

    fireEvent.wheel(viewport!, { deltaY: 100, deltaMode: 0 });
    await waitFor(() => expect(cameraInput).toHaveBeenCalledWith({ viewportId: 'viewport-1', zoom: -1 }));

    fireEvent.wheel(viewport!, { deltaY: -100, deltaMode: 0 });
    await waitFor(() => expect(cameraInput).toHaveBeenCalledWith({ viewportId: 'viewport-1', zoom: 1 }));

    expect(pointer).not.toHaveBeenCalledWith(expect.objectContaining({ phase: 'wheel' }));
  });
});
