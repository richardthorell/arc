import { describe, expect, it } from 'vitest';

import { panelRegistry } from '../app/panelRegistry';
import { getPanelTabPresentation } from './PanelDockTab';

describe('panel dock tab presentation', () => {
  it('uses panel registry title and icon by default', () => {
    const tab = getPanelTabPresentation('hierarchy');
    expect(tab.title).toBe('Hierarchy');
    expect(tab.icon).toBe(panelRegistry.hierarchy.icon);
    expect(tab.closeable).toBe(true);
  });

  it('keeps instance titles while reusing the panel icon', () => {
    const tab = getPanelTabPresentation('viewport', 'Viewport 3');
    expect(tab.title).toBe('Viewport 3');
    expect(tab.icon).toBe(panelRegistry.viewport.icon);
  });

  it('supports a panel registration without an icon', () => {
    const previous = panelRegistry.search.icon;
    panelRegistry.search.icon = null;
    try {
      expect(getPanelTabPresentation('search').icon).toBeNull();
    } finally {
      panelRegistry.search.icon = previous;
    }
  });
});
