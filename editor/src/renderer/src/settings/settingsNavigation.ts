import type { EditorSettingDescriptor } from '../../../common/editorWorkflowTypes';
import type { UiTreeNode } from '../ui';

export type EditorSettingsPageId =
  | 'general'
  | 'appearance'
  | 'editing.viewport'
  | 'editing.navigation'
  | 'editing.gizmos'
  | 'editing.scene'
  | 'content.browser'
  | 'content.import'
  | 'ai.providers'
  | 'ai.assistant'
  | 'ai.remote'
  | 'source-control'
  | 'platforms'
  | 'tools.external'
  | 'tools.shortcuts'
  | 'tools.extensions'
  | 'system.recovery'
  | 'system.performance'
  | 'system.cache'
  | 'system.diagnostics';

export type EditorSettingsPage = {
  id: EditorSettingsPageId;
  label: string;
  description: string;
  legacySection?: EditorSettingDescriptor['section'];
  keywords?: readonly string[];
};

export const editorSettingsPages: readonly EditorSettingsPage[] = [
  {
    id: 'general',
    label: 'General',
    description: 'Startup, project and general editor behavior.',
    legacySection: 'Editor',
    keywords: ['editor', 'startup', 'layout', 'project'],
  },
  { id: 'appearance', label: 'Appearance', description: 'Theme, scale and editor presentation.', keywords: ['theme', 'ui', 'scale'] },
  {
    id: 'editing.viewport',
    label: 'Viewport',
    description: 'Default viewport rendering and camera presentation.',
    legacySection: 'Renderer',
    keywords: ['renderer', 'render', 'camera', 'grid'],
  },
  {
    id: 'editing.navigation',
    label: 'Navigation',
    description: 'Mouse, keyboard and viewport navigation behavior.',
    legacySection: 'Input',
    keywords: ['input', 'mouse', 'keyboard', 'camera'],
  },
  { id: 'editing.gizmos', label: 'Gizmos & Snapping', description: 'Transform gizmos and snapping defaults.', keywords: ['transform', 'snap'] },
  { id: 'editing.scene', label: 'Scene', description: 'Scene editing behavior and defaults.', keywords: ['entity', 'selection'] },
  { id: 'content.browser', label: 'Content Browser', description: 'Asset browser defaults and presentation.', keywords: ['asset', 'thumbnail'] },
  { id: 'content.import', label: 'Asset Import', description: 'Default import and reimport behavior.', keywords: ['asset', 'import', 'reimport'] },
  { id: 'ai.providers', label: 'Providers', description: 'AI provider accounts and available models.', keywords: ['openai', 'anthropic', 'google', 'model'] },
  { id: 'ai.assistant', label: 'Assistant', description: 'Built-in AI assistant behavior and permissions.', keywords: ['prompt', 'agent', 'model'] },
  { id: 'ai.remote', label: 'Remote Access', description: 'Remote agent gateway access and permissions.', keywords: ['gateway', 'remote', 'agent'] },
  {
    id: 'source-control',
    label: 'Source Control',
    description: 'Version control provider and editor integration.',
    legacySection: 'Source Control',
    keywords: ['git', 'perforce', 'version control'],
  },
  { id: 'platforms', label: 'Platforms', description: 'Installed SDKs and platform toolchains.', keywords: ['android', 'windows', 'linux', 'apple', 'sdk'] },
  {
    id: 'tools.external',
    label: 'External Tools',
    description: 'External editors, terminals and tool paths.',
    legacySection: 'Paths & Tools',
    keywords: ['path', 'ide', 'terminal', 'diff'],
  },
  { id: 'tools.shortcuts', label: 'Keyboard Shortcuts', description: 'Editor command keybindings.', keywords: ['keyboard', 'shortcut', 'keybinding'] },
  {
    id: 'tools.extensions',
    label: 'Extensions',
    description: 'Project-declared editor extensions.',
    legacySection: 'Extensions',
    keywords: ['plugin', 'extension'],
  },
  {
    id: 'system.recovery',
    label: 'Auto Save & Recovery',
    description: 'Recovery generations and editor autosave behavior.',
    legacySection: 'Recovery',
    keywords: ['recovery', 'autosave', 'snapshot'],
  },
  { id: 'system.performance', label: 'Performance', description: 'Editor responsiveness and background work budgets.', keywords: ['fps', 'background', 'performance'] },
  {
    id: 'system.cache',
    label: 'Cache',
    description: 'Editor cache locations and behavior.',
    legacySection: 'Cache',
    keywords: ['derived data', 'disk', 'cache'],
  },
  { id: 'system.diagnostics', label: 'Diagnostics', description: 'Logging, crash reporting and developer diagnostics.', keywords: ['log', 'crash', 'diagnostic'] },
] as const;

const pageById = new Map<EditorSettingsPageId, EditorSettingsPage>(editorSettingsPages.map((page) => [page.id, page]));

export const getEditorSettingsPage = (id: string): EditorSettingsPage | null =>
  pageById.get(id as EditorSettingsPageId) ?? null;

const pageNode = (id: EditorSettingsPageId): UiTreeNode => {
  const page = pageById.get(id)!;
  return { id: page.id, label: page.label, keywords: page.keywords };
};

export const editorSettingsNavigation: readonly UiTreeNode[] = [
  pageNode('general'),
  pageNode('appearance'),
  {
    id: 'editing',
    label: 'Editing',
    keywords: ['viewport', 'navigation', 'gizmo', 'scene'],
    children: [pageNode('editing.viewport'), pageNode('editing.navigation'), pageNode('editing.gizmos'), pageNode('editing.scene')],
  },
  {
    id: 'content',
    label: 'Content',
    keywords: ['asset', 'import'],
    children: [pageNode('content.browser'), pageNode('content.import')],
  },
  {
    id: 'ai',
    label: 'AI',
    keywords: ['provider', 'assistant', 'gateway', 'remote'],
    children: [pageNode('ai.providers'), pageNode('ai.assistant'), pageNode('ai.remote')],
  },
  pageNode('source-control'),
  pageNode('platforms'),
  {
    id: 'tools',
    label: 'Tools',
    keywords: ['external', 'shortcut', 'extension'],
    children: [pageNode('tools.external'), pageNode('tools.shortcuts'), pageNode('tools.extensions')],
  },
  {
    id: 'system',
    label: 'System',
    keywords: ['recovery', 'performance', 'cache', 'diagnostics'],
    children: [pageNode('system.recovery'), pageNode('system.performance'), pageNode('system.cache'), pageNode('system.diagnostics')],
  },
] as const;

export const defaultExpandedSettingsNodes = ['editing', 'content', 'ai', 'tools', 'system'] as const;
