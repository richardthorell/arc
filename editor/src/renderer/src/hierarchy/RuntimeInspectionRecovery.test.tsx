// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { LIVE_PLAY_PROTOCOL, LIVE_PLAY_VERSION, type LivePlayMessage } from '../../../common/livePlayProtocol';
import type { LivePlayHierarchyState } from '../../../common/livePlayHierarchyState';
import type { WindowsLivePlaySnapshot } from '../../../common/livePlaySessionTypes';
import { useLivePlayRuntimeInspection } from './RuntimeInspectionPanels';

const session: WindowsLivePlaySnapshot = {
  state: 'connected',
  sessionId: 'session-1',
  instanceId: 'runtime-1',
  pid: 42,
  lastReceivedSequence: 2,
  error: null,
};
const authority = {
  protocol: LIVE_PLAY_PROTOCOL,
  protocolVersion: LIVE_PLAY_VERSION,
  sender: 'runtime' as const,
  projectGuid: '11111111-1111-1111-1111-111111111111',
  buildId: 'dev-build',
  targetProfile: 'windows-default',
  sessionId: 'session-1',
  instanceId: 'runtime-1',
  messageId: 'message-1',
  sequence: 1,
} as const;
const runtimeEntity = { kind: 'runtime' as const, runtimeEntityId: 'w1-x1-e4-g1' };
const base = {
  ...authority,
  type: 'hierarchy.snapshot' as const,
  payload: {
    worldRevision: 3,
    entities: [{ entity: runtimeEntity, parent: null, name: 'Old' }],
    complete: true,
  },
};
const next = {
  ...authority,
  messageId: 'message-2',
  sequence: 2,
  type: 'hierarchy.changed' as const,
  payload: {
    baseRevision: 3,
    worldRevision: 4,
    changes: [{ kind: 'upsert' as const, value: { entity: runtimeEntity, parent: null, name: 'Renamed' } }],
  },
};

afterEach(() => {
  cleanup();
  Object.defineProperty(window, 'arc', { value: undefined, configurable: true });
});

describe('Live Play reconnect renderer hydration', () => {
  it('replays early revisioned hierarchy changes after loading the verified base', async () => {
    const handlers: {
      notify?: (message: LivePlayMessage) => void;
      resolveCache?: (state: LivePlayHierarchyState) => void;
    } = {};
    const cache = new Promise<LivePlayHierarchyState>((resolve) => {
      handlers.resolveCache = resolve;
    });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        livePlay: {
          onMessage: (listener: (message: LivePlayMessage) => void) => {
            handlers.notify = listener;
            return () => {
              handlers.notify = undefined;
            };
          },
          latestHierarchyState: () => cache,
          inspect: async () => null,
        },
      },
    });
    const { result } = renderHook(() => useLivePlayRuntimeInspection(session));
    await act(async () => {
      handlers.notify?.(next);
    });
    expect(result.current.hierarchy).toBeNull();
    await act(async () => {
      handlers.resolveCache?.({ hierarchy: base, pending: null, requiresSnapshot: false });
      await cache;
    });
    await waitFor(() => {
      expect(result.current.hierarchy?.payload.worldRevision).toBe(4);
      expect(result.current.hierarchy?.payload.entities[0].name).toBe('Renamed');
      expect(result.current.requiresSnapshot).toBe(false);
    });
  });

  it('retains a stale but verified hierarchy on reload while waiting for full recovery', async () => {
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        livePlay: {
          onMessage: () => () => {},
          latestHierarchyState: async () => ({ hierarchy: base, pending: null, requiresSnapshot: true }),
          inspect: async () => null,
        },
      },
    });
    const { result } = renderHook(() => useLivePlayRuntimeInspection({ ...session, state: 'disconnected' }));
    await waitFor(() => {
      expect(result.current.hierarchy?.payload.worldRevision).toBe(3);
      expect(result.current.requiresSnapshot).toBe(true);
      expect(result.current.properties).toEqual({});
    });
  });
});
