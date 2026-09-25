// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { defaultWorkbenchLayout, useWorkbenchLayout } from './workbenchStore';

const layoutStorageKey = 'arc.editor.workbench.layout.v2';

beforeEach(() => window.localStorage.clear());
afterEach(cleanup);

describe('useWorkbenchLayout', () => {
  it('always boots with utility drawers closed and does not persist their open state', () => {
    window.localStorage.setItem(
      layoutStorageKey,
      JSON.stringify({
        ...defaultWorkbenchLayout,
        activeActivity: 'search',
        activityExpanded: true,
      }),
    );

    const { result } = renderHook(() => useWorkbenchLayout());
    expect(result.current.layout.activeActivity).toBe('search');
    expect(result.current.layout.activityExpanded).toBe(false);

    act(() => {
      result.current.setLayout((current) => ({ ...current, activityExpanded: true }));
    });

    const saved = JSON.parse(window.localStorage.getItem(layoutStorageKey) ?? '{}') as Record<string, unknown>;
    expect(saved.activeActivity).toBe('search');
    expect(saved).not.toHaveProperty('activityExpanded');
  });
});
