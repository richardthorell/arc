import type { AiProviderId } from '../../../common/aiProviderTypes';
import type { EditorSettingDescriptor } from '../../../common/editorWorkflowTypes';
import type { UiTreeNode } from '../ui';
import generalSettingsHeader from './assets/general-settings-header.webp';
import viewportSettingsHeader from './assets/viewport-settings-header.webp';

export type EditorSettingsPageId =
  | 'general'
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
  | 'platforms.windows'
  | 'tools.external'
  | 'tools.shortcuts'
  | 'tools.extensions'
  | 'system.recovery'
  | 'system.performance'
  | 'system.cache'
  | 'system.diagnostics';

export type EditorSettingsContentKind = 'settings' | 'workbench' | 'recovery' | 'extensions';
export type EditorSettingsIcon = 'palette' | 'viewport' | 'openai' | 'anthropic';

export type EditorSettingsCardDefinition = {
  section: EditorSettingDescriptor['section'];
  title?: string;
  icon?: EditorSettingsIcon;
  provider?: AiProviderId;
};

export type EditorSettingsPage = {
  id: EditorSettingsPageId;
  label: string;
  description: string;
  keywords?: readonly string[];
  headerImage?: string;
  cards?: readonly EditorSettingsCardDefinition[];
  content?: readonly EditorSettingsContentKind[];
};

type EditorSettingsGroup = {
  id: string;
  label: string;
  keywords?: readonly string[];
  defaultExpanded?: boolean;
  children: readonly EditorSettingsDefinitionNode[];
};

export type EditorSettingsDefinitionNode = EditorSettingsPage | EditorSettingsGroup;

export const editorSettingsDefinition: readonly EditorSettingsDefinitionNode[] = [
  {
    id: 'general',
    label: 'General',
    description: 'Appearance, startup and general editor behavior.',
    keywords: ['editor', 'startup', 'layout', 'project', 'appearance', 'theme', 'ui', 'scale'],
    headerImage: generalSettingsHeader,
    cards: [{ section: 'Editor', title: 'Appearance', icon: 'palette' }],
    content: ['settings', 'workbench'],
  },
  {
    id: 'editing',
    label: 'Editing',
    keywords: ['viewport', 'navigation', 'gizmo', 'scene'],
    defaultExpanded: true,
    children: [
      {
        id: 'editing.viewport',
        label: 'Viewport',
        description: 'Default viewport rendering and camera presentation.',
        keywords: ['renderer', 'render', 'camera', 'grid'],
        headerImage: viewportSettingsHeader,
        cards: [{ section: 'Renderer', title: 'Viewport Rendering', icon: 'viewport' }],
      },
      {
        id: 'editing.navigation',
        label: 'Navigation',
        description: 'Mouse, keyboard and viewport navigation behavior.',
        keywords: ['input', 'mouse', 'keyboard', 'camera'],
      },
      {
        id: 'editing.gizmos',
        label: 'Gizmos & Snapping',
        description: 'Transform gizmos and snapping defaults.',
        keywords: ['transform', 'snap', 'translation', 'rotation', 'scale'],
        cards: [{ section: 'Input' }],
      },
      {
        id: 'editing.scene',
        label: 'Scene',
        description: 'Scene editing behavior and defaults.',
        keywords: ['entity', 'selection'],
      },
    ],
  },
  {
    id: 'content',
    label: 'Content',
    keywords: ['asset', 'import'],
    defaultExpanded: true,
    children: [
      {
        id: 'content.browser',
        label: 'Content Browser',
        description: 'Asset browser defaults and presentation.',
        keywords: ['asset', 'thumbnail'],
      },
      {
        id: 'content.import',
        label: 'Asset Import',
        description: 'Default import and reimport behavior.',
        keywords: ['asset', 'import', 'reimport'],
      },
    ],
  },
  {
    id: 'ai',
    label: 'AI',
    keywords: ['provider', 'assistant', 'gateway', 'remote'],
    defaultExpanded: true,
    children: [
      {
        id: 'ai.providers',
        label: 'Providers',
        description: 'AI provider accounts and available models.',
        cards: [
          { section: 'OpenAI', title: 'OpenAI', icon: 'openai', provider: 'openai' },
          { section: 'Anthropic', title: 'Anthropic', icon: 'anthropic', provider: 'anthropic' },
        ],
        keywords: ['openai', 'anthropic', 'claude', 'api key', 'model', 'reasoning', 'effort'],
      },
      {
        id: 'ai.assistant',
        label: 'Assistant',
        description: 'Built-in AI assistant behavior and permissions.',
        keywords: ['prompt', 'agent', 'model'],
      },
      {
        id: 'ai.remote',
        label: 'Remote Agent Access',
        description: 'Remote agent gateway access and permissions.',
        keywords: ['gateway', 'remote', 'agent'],
      },
    ],
  },
  {
    id: 'source-control',
    label: 'Source Control',
    description: 'Version control provider and editor integration.',
    keywords: ['git', 'perforce', 'version control'],
    cards: [{ section: 'Source Control' }],
  },
  {
    id: 'platforms',
    label: 'Platforms & SDKs',
    keywords: ['windows', 'android', 'linux', 'apple', 'sdk', 'toolchain'],
    defaultExpanded: true,
    children: [
      {
        id: 'platforms.windows',
        label: 'Windows',
        description: 'Windows SDK and native build toolchain locations.',
        keywords: ['windows', 'msvc', 'visual studio', 'sdk', 'cmake', 'ninja', 'compiler', 'toolchain'],
        cards: [{ section: 'Windows' }],
      },
    ],
  },
  {
    id: 'tools',
    label: 'Tools',
    keywords: ['external', 'shortcut', 'extension'],
    defaultExpanded: true,
    children: [
      {
        id: 'tools.external',
        label: 'External Tools',
        description: 'External editors, terminals and tool paths.',
        keywords: ['path', 'ide', 'terminal', 'diff'],
        cards: [{ section: 'Paths & Tools' }],
      },
      {
        id: 'tools.shortcuts',
        label: 'Keyboard Shortcuts',
        description: 'Editor command keybindings.',
        keywords: ['keyboard', 'shortcut', 'keybinding'],
      },
      {
        id: 'tools.extensions',
        label: 'Extensions',
        description: 'Project-declared editor extensions.',
        keywords: ['plugin', 'extension'],
        cards: [{ section: 'Extensions' }],
        content: ['settings', 'extensions'],
      },
    ],
  },
  {
    id: 'system',
    label: 'System',
    keywords: ['recovery', 'performance', 'cache', 'diagnostics', 'logging'],
    defaultExpanded: true,
    children: [
      {
        id: 'system.recovery',
        label: 'Auto Save & Recovery',
        description: 'Recovery generations and editor autosave behavior.',
        keywords: ['recovery', 'autosave', 'snapshot'],
        cards: [{ section: 'Recovery' }],
        content: ['settings', 'recovery'],
      },
      {
        id: 'system.performance',
        label: 'Performance',
        description: 'Editor responsiveness and background work budgets.',
        keywords: ['fps', 'background', 'performance'],
      },
      {
        id: 'system.cache',
        label: 'Cache',
        description: 'Editor cache locations and behavior.',
        keywords: ['derived data', 'disk', 'cache'],
        cards: [{ section: 'Cache' }],
      },
      {
        id: 'system.diagnostics',
        label: 'Diagnostics & Logging',
        description: 'Logging, crash reporting and developer diagnostics.',
        keywords: ['log', 'logging', 'crash', 'diagnostic'],
      },
    ],
  },
] as const;

const isGroup = (node: EditorSettingsDefinitionNode): node is EditorSettingsGroup => 'children' in node;

const flattenPages = (nodes: readonly EditorSettingsDefinitionNode[]): EditorSettingsPage[] =>
  nodes.flatMap((node) => (isGroup(node) ? flattenPages(node.children) : [node]));

const toNavigationNode = (node: EditorSettingsDefinitionNode): UiTreeNode => ({
  id: node.id,
  label: node.label,
  keywords: node.keywords,
  children: isGroup(node) ? node.children.map(toNavigationNode) : undefined,
});

const collectDefaultExpandedIds = (nodes: readonly EditorSettingsDefinitionNode[]): string[] =>
  nodes.flatMap((node) => {
    if (!isGroup(node)) return [];
    return [...(node.defaultExpanded ? [node.id] : []), ...collectDefaultExpandedIds(node.children)];
  });

export const editorSettingsPages = flattenPages(editorSettingsDefinition);

const pageById = new Map<EditorSettingsPageId, EditorSettingsPage>(
  editorSettingsPages.map((page) => [page.id, page] as const),
);

export const getEditorSettingsPage = (id: string): EditorSettingsPage | null =>
  pageById.get(id as EditorSettingsPageId) ?? null;

export const editorSettingsNavigation: readonly UiTreeNode[] = editorSettingsDefinition.map(toNavigationNode);

export const defaultExpandedSettingsNodes = collectDefaultExpandedIds(editorSettingsDefinition);
