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
      'platforms.build-tools',
      'platforms.windows',
      'platforms.android',
      'platforms.linux',
      'platforms.apple.xcode',
      'platforms.apple.macos',
      'platforms.apple.ios',
      'platforms.web',
      'platforms.xbox',
      'platforms.playstation',
      'platforms.switch',
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
    expect(editorSettingsNavigation.find((node) => node.id === 'platforms')?.children?.map((node) => node.id)).toEqual([
      'platforms.build-tools',
      'platforms.windows',
      'platforms.android',
      'platforms.linux',
      'platforms.apple',
      'platforms.web',
      'platforms.xbox',
      'platforms.playstation',
      'platforms.switch',
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
    expect(getEditorSettingsPage('ai.providers')?.cards).toEqual([
      { section: 'OpenAI', title: 'OpenAI', icon: 'openai', provider: 'openai' },
      { section: 'Anthropic', title: 'Anthropic', icon: 'anthropic', provider: 'anthropic' },
    ]);
    expect(getEditorSettingsPage('platforms.build-tools')?.cards?.[0]).toMatchObject({
      section: 'Windows',
      title: 'Build Tools',
      icon: 'hammer',
      keys: ['platform.windows.cmakePath', 'platform.windows.ninjaPath'],
    });
    expect(getEditorSettingsPage('platforms.windows')?.cards?.[0]).toMatchObject({ icon: 'windows' });
    expect(getEditorSettingsPage('platforms.android')?.cards?.[0]).toMatchObject({ icon: 'android' });
    expect(getEditorSettingsPage('platforms.linux')?.cards?.[0]).toMatchObject({ icon: 'linux' });
    expect(getEditorSettingsPage('platforms.apple.ios')?.cards).toEqual([{ section: 'iOS', icon: 'apple' }]);
    expect(getEditorSettingsPage('platforms.web')?.cards?.[0]).toMatchObject({ icon: 'web' });
    expect(getEditorSettingsPage('platforms.xbox')?.cards?.[0]).toMatchObject({ icon: 'xbox' });
    expect(getEditorSettingsPage('platforms.playstation')?.cards?.[0]).toMatchObject({ icon: 'playstation' });
    expect(getEditorSettingsPage('platforms.switch')?.cards?.[0]).toMatchObject({ icon: 'switch' });
    expect(getEditorSettingsPage('system.recovery')?.content).toEqual(['settings', 'recovery']);
    expect(getEditorSettingsPage('tools.extensions')?.content).toEqual(['settings', 'extensions']);
  });

  it('inherits group header images into child pages', () => {
    for (const pageId of ['editing.viewport', 'editing.navigation', 'editing.gizmos', 'editing.scene']) {
      expect(getEditorSettingsPage(pageId)?.headerImage).toContain('editing-settings-header');
    }
    expect(getEditorSettingsPage('platforms.windows')?.headerImage).toContain('platforms-settings-header');
    expect(getEditorSettingsPage('platforms.android')?.headerImage).toContain('platforms-settings-header');
  });

  it('derives default-expanded groups from the hierarchy', () => {
    expect(defaultExpandedSettingsNodes).toEqual([
      'editing',
      'content',
      'ai',
      'platforms',
      'platforms.apple',
      'tools',
      'system',
    ]);
    expect(getEditorSettingsPage('editing')).toBeNull();
    expect(getEditorSettingsPage('platforms.apple')).toBeNull();
  });
});
