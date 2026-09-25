import { describe, expect, it } from 'vitest';

import { activityRegistry, dockPanelIds, isSidebarPanel, panelRegistry, sidebarPanelIds } from './panelRegistry';

describe('primary sidebar registry', () => {
  it('retains the hierarchy activity for layout compatibility and exposes three panel utilities', () => {
    expect(activityRegistry.map((activity) => activity.panelId)).toEqual([
      'hierarchy',
      'search',
      'aiAssistant',
      'versionControl',
    ]);
    expect([...sidebarPanelIds]).toEqual(['search', 'aiAssistant', 'versionControl']);
    expect('settings' in panelRegistry).toBe(false);
  });

  it('keeps workspace tools out of the primary sidebar', () => {
    expect(isSidebarPanel('hierarchy')).toBe(false);
    expect(isSidebarPanel('search')).toBe(true);
    expect(isSidebarPanel('aiAssistant')).toBe(true);
    expect(isSidebarPanel('versionControl')).toBe(true);
    expect(isSidebarPanel('viewport')).toBe(false);
    expect(isSidebarPanel('inspector')).toBe(false);
    expect(isSidebarPanel('contentBrowser')).toBe(false);
    expect(isSidebarPanel('profiler')).toBe(false);
  });

  it('uses Inspector as the only right-side details panel', () => {
    expect(dockPanelIds.right).toEqual(['inspector']);
    expect('lighting' in panelRegistry).toBe(false);
    expect('worldSettings' in panelRegistry).toBe(false);
  });
});
