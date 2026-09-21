// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { EditorSurfaceActivityProvider } from '../editors/EditorSurfaceActivity';
import { AssetPreviewViewport } from './AssetPreviewViewport';

class TestResizeObserver {
  observe() {}
  disconnect() {}
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

describe('AssetPreviewViewport visibility', () => {
  it('keeps a preview surface alive while its document is inactive', async () => {
    const create = vi.fn().mockResolvedValue({ succeeded: true });
    const detach = vi.fn().mockResolvedValue({ succeeded: true });
    const setVisibility = vi.fn().mockResolvedValue({ succeeded: true });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        getStartupState: vi.fn().mockResolvedValue({ engineHostConnected: true, viewportMode: 'streamed' }),
        host: { query: vi.fn().mockResolvedValue({ succeeded: true, payload: {} }) },
        viewport: {
          create,
          resize: vi.fn().mockResolvedValue({ succeeded: true }),
          detach,
          setVisibility,
          registerSurface: vi.fn(),
          unregisterSurface: vi.fn(),
        },
      },
    });

    const preview = (
      <AssetPreviewViewport kind="shader" assetGuid="shader-guid" fallback="Unavailable" label="Preview" />
    );
    const view = render(<EditorSurfaceActivityProvider active>{preview}</EditorSurfaceActivityProvider>);
    await waitFor(() => expect(create).toHaveBeenCalledTimes(1));
    view.rerender(<EditorSurfaceActivityProvider active={false}>{preview}</EditorSurfaceActivityProvider>);
    await waitFor(() => expect(setVisibility).toHaveBeenCalledWith(expect.any(String), false));
    expect(detach).not.toHaveBeenCalled();

    view.rerender(<EditorSurfaceActivityProvider active>{preview}</EditorSurfaceActivityProvider>);
    await waitFor(() => expect(setVisibility).toHaveBeenLastCalledWith(expect.any(String), true));
    expect(create).toHaveBeenCalledTimes(1);
    view.unmount();
    await waitFor(() => expect(detach).toHaveBeenCalledTimes(1));
  });
});
