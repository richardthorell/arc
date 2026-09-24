import { describe, expect, it } from 'vitest';

import {
  defaultExpandedSettingsNodes,
  editorSettingsDefinition,
  editorSettingsNavigation,
  editorSettingsPages,
  getEditorSettingsPage,
} from './settingsNavigation';

describe('settingsNavigation', () => {
  it('derives pages and navigation from the same definition tree', () => {
    expect(editorSettingsDefinition).toHaveLength(editorSettingsNavigation.length);
    expect(editorSettingsPages.map((page) => page.id)).toEqual([
      'general',
      'editing.viewport',
      'editing.navigation',
      'editing.gizmos',
      'editing.scene',
      'content.browser',
      'content.import',
      'ai.providers',
      'ai.assistant',
      'ai.remote',
      'source-control',
      'platforms.windows',
      'tools.external',
      'tools.shortcuts',
      'tools.extensions',
      'system.recovery',
      'system.performance',
      'system.cache',
      'system.diagnostics',
    ]);

    expect(editorSettingsNavigation.find((node) => node.id === 'editing')?.children?.map((node) => node.id)).toEqual([
      'editing.viewport',
      'editing.navigation',
      'editing.gizmos',
      'editing.scene',
    ]);
  });

  it('keeps card presentation and supplemental content in page metadata', () => {
    expect(getEditorSettingsPage('general')).toMatchObject({
      cards: [{ section: 'Editor', title: 'Appearance', icon: 'palette' }],
      content: ['settings', 'workbench'],
    });
    expect(getEditorSettingsPage('general')?.headerImage).toContain('general-settings-header');
    expect(getEditorSettingsPage('editing.viewport')).toMatchObject({
      cards: [{ section: 'Renderer', title: 'Viewport Rendering', icon: 'viewport' }],
    });
    expect(getEditorSettingsPage('editing.viewport')?.headerImage).toContain('viewport-settings-header');
    expect(getEditorSettingsPage('ai.providers')?.cards).toEqual([
      { section: 'OpenAI', title: 'OpenAI', icon: 'openai', provider: 'openai' },
      { section: 'Anthropic', title: 'Anthropic', icon: 'anthropic', provider: 'anthropic' },
    ]);
    expect(getEditorSettingsPage('system.recovery')?.content).toEqual(['settings', 'recovery']);
    expect(getEditorSettingsPage('tools.extensions')?.content).toEqual(['settings', 'extensions']);
  });

  it('derives default-expanded groups from the hierarchy', () => {
    expect(defaultExpandedSettingsNodes).toEqual(['editing', 'content', 'ai', 'platforms', 'tools', 'system']);
    expect(getEditorSettingsPage('editing')).toBeNull();
  });
});
