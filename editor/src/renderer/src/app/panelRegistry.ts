import {
  Box,
  Bot,
  Database,
  FileCode2,
  FileText,
  FolderTree,
  Gauge,
  GitBranch,
  Layers3,
  Lightbulb,
  Search,
  Settings,
  SlidersHorizontal,
  Globe2,
} from 'lucide-react';

import type { ActivityRegistration, PanelRegistration, WorkbenchPanelId } from './workbenchTypes';

// Shared panel metadata. Dockview consumes this for dockable tab presentation,
// while the primary sidebar reuses the same title/icon metadata for its fixed
// activity-hosted panels.
export const panelRegistry: Record<WorkbenchPanelId, PanelRegistration> = {
  hierarchy: {
    id: 'hierarchy',
    title: 'Hierarchy',
    icon: FolderTree,
    defaultRegion: 'left',
    activityId: 'scene',
    minimumWidth: 220,
    closeable: true,
  },
  assetExplorer: { id: 'assetExplorer', title: 'Assets', icon: Database, defaultRegion: 'left', activityId: 'assets' },
  search: { id: 'search', title: 'Search', icon: Search, defaultRegion: 'left', activityId: 'search' },
  viewport: {
    id: 'viewport',
    title: 'Viewport 1',
    icon: Box,
    defaultRegion: 'center',
    allowMultiple: true,
    minimumWidth: 360,
    minimumHeight: 240,
    closeable: true,
  },
  renderGraph: {
    id: 'renderGraph',
    title: 'Render Graph',
    icon: Layers3,
    defaultRegion: 'center',
    activityId: 'renderGraph',
  },
  shaderEditor: { id: 'shaderEditor', title: 'pbr_lit.hlsl', icon: FileCode2, defaultRegion: 'center' },
  inspector: {
    id: 'inspector',
    title: 'Inspector',
    icon: SlidersHorizontal,
    defaultRegion: 'right',
    minimumWidth: 320,
    closeable: true,
  },
  lighting: { id: 'lighting', title: 'Lighting', icon: Lightbulb, defaultRegion: 'right' },
  worldSettings: { id: 'worldSettings', title: 'World Settings', icon: Globe2, defaultRegion: 'right' },
  contentBrowser: {
    id: 'contentBrowser',
    title: 'Content Browser',
    icon: Database,
    defaultRegion: 'bottom',
    activityId: 'assets',
  },
  console: { id: 'console', title: 'Console', icon: FileText, defaultRegion: 'bottom' },
  buildOutput: { id: 'buildOutput', title: 'Build Output', icon: FileCode2, defaultRegion: 'bottom' },
  versionControl: {
    id: 'versionControl',
    title: 'Version Control',
    icon: GitBranch,
    defaultRegion: 'bottom',
    activityId: 'versionControl',
  },
  aiAssistant: {
    id: 'aiAssistant',
    title: 'AI Gateway',
    icon: Bot,
    defaultRegion: 'bottom',
    activityId: 'aiAssistant',
  },
  profiler: { id: 'profiler', title: 'Profiler', icon: Gauge, defaultRegion: 'bottom', activityId: 'profiler' },
  settings: { id: 'settings', title: 'Settings', icon: Settings, defaultRegion: 'left', activityId: 'settings' },
};

export const sidebarPanelIds = ['hierarchy', 'search', 'aiAssistant', 'versionControl'] as const;
export type SidebarPanelId = (typeof sidebarPanelIds)[number];

export const isSidebarPanel = (panel: WorkbenchPanelId): panel is SidebarPanelId =>
  (sidebarPanelIds as readonly WorkbenchPanelId[]).includes(panel);

// The activity rail is intentionally small and stable. It switches the fixed
// primary sidebar only; it does not rearrange or activate Dockview groups.
export const activityRegistry: ActivityRegistration[] = [
  { id: 'scene', title: 'Hierarchy', icon: FolderTree, panelId: 'hierarchy' },
  { id: 'search', title: 'Search', icon: Search, panelId: 'search' },
  { id: 'aiAssistant', title: 'AI Gateway', icon: Bot, panelId: 'aiAssistant' },
  { id: 'versionControl', title: 'Version Control', icon: GitBranch, panelId: 'versionControl' },
];

export const dockPanelIds = {
  center: ['viewport', 'renderGraph', 'shaderEditor'] satisfies WorkbenchPanelId[],
  right: ['inspector', 'lighting', 'worldSettings'] satisfies WorkbenchPanelId[],
  bottom: [
    'contentBrowser',
    'console',
    'buildOutput',
    'versionControl',
    'profiler',
    'aiAssistant',
  ] satisfies WorkbenchPanelId[],
};

export const getPanel = (id: WorkbenchPanelId) => panelRegistry[id];
