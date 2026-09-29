import type { AiProviderId } from '../../../common/aiProviderTypes';
import type { EditorSettingDescriptor } from '../../../common/editorWorkflowTypes';
import type { UiTreeNode } from '../ui';
import generalSettingsHeader from './assets/general-settings-header.webp';
import platformsSettingsHeader from './assets/platforms-settings-header.webp';
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
  | 'platforms.build-tools'
  | 'platforms.windows'
  | 'platforms.android'
  | 'platforms.linux'
  | 'platforms.apple.xcode'
  | 'platforms.apple.macos'
  | 'platforms.apple.ios'
  | 'platforms.apple.tvos'
  | 'platforms.apple.visionos'
  | 'platforms.web'
  | 'platforms.xbox'
  | 'platforms.playstation'
  | 'platforms.switch'
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
  keys?: readonly string[];
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
  headerImage?: string;
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
    keywords: ['windows', 'android', 'linux', 'apple', 'ios', 'macos', 'web', 'sdk', 'toolchain', 'wsl'],
    headerImage: platformsSettingsHeader,
    defaultExpanded: true,
    children: [
      {
        id: 'platforms.build-tools',
        label: 'Build Tools',
        description: 'Shared native build tools used across ARC target platforms.',
        keywords: ['cmake', 'ninja', 'build'],
        cards: [
          {
            section: 'Windows',
            title: 'Build Tools',
            keys: ['platform.windows.cmakePath', 'platform.windows.ninjaPath'],
          },
        ],
      },
      {
        id: 'platforms.windows',
        label: 'Windows',
        description: 'Visual Studio, MSVC and Windows SDK configuration for local Windows builds.',
        keywords: ['windows', 'msvc', 'visual studio', 'sdk', 'compiler', 'toolchain'],
        cards: [
          {
            section: 'Windows',
            keys: [
              'platform.windows.visualStudioPath',
              'platform.windows.msvcToolchainPath',
              'platform.windows.sdkPath',
            ],
          },
        ],
      },
      {
        id: 'platforms.android',
        label: 'Android',
        description: 'Java, Android SDK and NDK locations used by Android builds and device tooling.',
        keywords: ['android', 'java', 'jdk', 'sdk', 'ndk', 'adb', 'toolchain'],
        cards: [{ section: 'Android', title: 'Android Toolchain' }],
      },
      {
        id: 'platforms.linux',
        label: 'Linux',
        description: 'Build Linux locally on Linux or through WSL when ARC runs on Windows.',
        keywords: ['linux', 'wsl', 'clang', 'gcc', 'sysroot', 'compiler'],
        cards: [{ section: 'Linux', title: 'Linux Build Environment' }],
      },
      {
        id: 'platforms.apple',
        label: 'Apple',
        keywords: ['apple', 'xcode', 'macos', 'ios', 'tvos', 'visionos'],
        defaultExpanded: true,
        children: [
          {
            id: 'platforms.apple.xcode',
            label: 'Xcode',
            description: 'Xcode developer tools used for all local Apple platform builds.',
            keywords: ['xcode', 'developer dir'],
            cards: [{ section: 'Apple', title: 'Apple Toolchain' }],
          },
          {
            id: 'platforms.apple.macos',
            label: 'macOS',
            description: 'macOS SDK derived from the selected Xcode installation.',
            keywords: ['macos', 'sdk'],
            cards: [{ section: 'macOS' }],
          },
          {
            id: 'platforms.apple.ios',
            label: 'iOS',
            description: 'iOS device and Simulator SDKs derived from Xcode.',
            keywords: ['ios', 'iphone', 'simulator', 'sdk'],
            cards: [{ section: 'iOS' }],
          },
          {
            id: 'platforms.apple.tvos',
            label: 'tvOS',
            description: 'tvOS device and Simulator SDKs derived from Xcode.',
            keywords: ['tvos', 'apple tv', 'simulator', 'sdk'],
            cards: [{ section: 'tvOS' }],
          },
          {
            id: 'platforms.apple.visionos',
            label: 'visionOS',
            description: 'visionOS device and Simulator SDKs derived from Xcode.',
            keywords: ['visionos', 'vision pro', 'simulator', 'sdk'],
            cards: [{ section: 'visionOS' }],
          },
        ],
      },
      {
        id: 'platforms.web',
        label: 'Web',
        description: 'Emscripten SDK used for WebAssembly builds.',
        keywords: ['web', 'webassembly', 'wasm', 'emscripten', 'emsdk'],
        cards: [{ section: 'Web' }],
      },
      {
        id: 'platforms.xbox',
        label: 'Xbox',
        description: 'Licensed Microsoft GDK root for Xbox development.',
        keywords: ['xbox', 'gdk', 'console'],
        cards: [{ section: 'Xbox' }],
      },
      {
        id: 'platforms.playstation',
        label: 'PlayStation',
        description: 'Licensed PlayStation SDK root for console development.',
        keywords: ['playstation', 'ps5', 'sdk', 'console'],
        cards: [{ section: 'PlayStation' }],
      },
      {
        id: 'platforms.switch',
        label: 'Nintendo Switch',
        description: 'Licensed Nintendo SDK root for Switch development.',
        keywords: ['nintendo', 'switch', 'sdk', 'console'],
        cards: [{ section: 'Nintendo Switch' }],
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

const flattenPages = (
  nodes: readonly EditorSettingsDefinitionNode[],
  inheritedHeaderImage?: string,
): EditorSettingsPage[] =>
  nodes.flatMap((node) => {
    if (isGroup(node)) return flattenPages(node.children, node.headerImage ?? inheritedHeaderImage);
    if (node.headerImage || !inheritedHeaderImage) return [node];
    return [{ ...node, headerImage: inheritedHeaderImage }];
  });

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
