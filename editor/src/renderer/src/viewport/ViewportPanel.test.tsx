// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ViewportPanel } from './ViewportPanel';

class TestResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', TestResizeObserver);
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
});
