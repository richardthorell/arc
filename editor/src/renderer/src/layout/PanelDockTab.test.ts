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

  it('normalizes viewport instance ids back to the viewport presentation', () => {
    const tab = getPanelTabPresentation('viewport-3', 'Viewport 3');
    expect(tab.title).toBe('Viewport 3');
    expect(tab.icon).toBe(panelRegistry.viewport.icon);
  });

  it('falls back cleanly for a component without registered panel metadata', () => {
    const tab = getPanelTabPresentation('extension-panel', 'Extension Panel');
    expect(tab.title).toBe('Extension Panel');
    expect(tab.icon).toBeNull();
    expect(tab.closeable).toBe(true);
  });
});
