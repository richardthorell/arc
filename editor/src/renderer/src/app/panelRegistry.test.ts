import { describe, expect, it } from 'vitest';

import { activityRegistry, isSidebarPanel, sidebarPanelIds } from './panelRegistry';

describe('primary sidebar registry', () => {
  it('contains only the four fixed sidebar activities', () => {
    expect(activityRegistry.map((activity) => activity.panelId)).toEqual([
      'hierarchy',
      'search',
      'aiAssistant',
      'versionControl',
    ]);
    expect([...sidebarPanelIds]).toEqual(['hierarchy', 'search', 'aiAssistant', 'versionControl']);
  });

  it('keeps workspace tools out of the primary sidebar', () => {
    expect(isSidebarPanel('hierarchy')).toBe(true);
    expect(isSidebarPanel('search')).toBe(true);
    expect(isSidebarPanel('aiAssistant')).toBe(true);
    expect(isSidebarPanel('versionControl')).toBe(true);
    expect(isSidebarPanel('viewport')).toBe(false);
    expect(isSidebarPanel('inspector')).toBe(false);
    expect(isSidebarPanel('contentBrowser')).toBe(false);
    expect(isSidebarPanel('profiler')).toBe(false);
  });
});
