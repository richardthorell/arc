import {
  Box,
  Bot,
  Bug,
  Database,
  FileCode2,
  FileText,
  FolderTree,
  Gauge,
  GitBranch,
  Layers3,
  Lightbulb,
  Package,
  Search,
  Settings,
  SlidersHorizontal,
} from 'lucide-react';

import type { ActivityRegistration, PanelRegistration, WorkbenchPanelId } from './workbenchTypes';

export const panelRegistry: Record<WorkbenchPanelId, PanelRegistration> = {
  hierarchy: { id: 'hierarchy', title: 'Hierarchy', icon: FolderTree, defaultRegion: 'left', activityId: 'scene' },
  assetExplorer: { id: 'assetExplorer', title: 'Assets', icon: Database, defaultRegion: 'left', activityId: 'assets' },
  search: { id: 'search', title: 'Search', icon: Search, defaultRegion: 'left', activityId: 'search' },
  viewport: { id: 'viewport', title: 'Viewport 1', icon: Box, defaultRegion: 'center' },
  renderGraph: { id: 'renderGraph', title: 'Render Graph', icon: Layers3, defaultRegion: 'center', activityId: 'renderGraph' },
  shaderEditor: { id: 'shaderEditor', title: 'pbr_lit.hlsl', icon: FileCode2, defaultRegion: 'center' },
  inspector: { id: 'inspector', title: 'Inspector', icon: SlidersHorizontal, defaultRegion: 'right' },
  lighting: { id: 'lighting', title: 'Lighting', icon: Lightbulb, defaultRegion: 'right' },
  worldSettings: { id: 'worldSettings', title: 'World Settings', icon: Settings, defaultRegion: 'right' },
  contentBrowser: { id: 'contentBrowser', title: 'Content Browser', icon: Database, defaultRegion: 'bottom', activityId: 'assets' },
  console: { id: 'console', title: 'Console', icon: FileText, defaultRegion: 'bottom' },
  versionControl: { id: 'versionControl', title: 'Version Control', icon: GitBranch, defaultRegion: 'bottom', activityId: 'versionControl' },
  aiAssistant: { id: 'aiAssistant', title: 'AI Assistant', icon: Bot, defaultRegion: 'bottom', activityId: 'aiAssistant' },
  profiler: { id: 'profiler', title: 'Profiler', icon: Gauge, defaultRegion: 'bottom', activityId: 'profiler' },
  settings: { id: 'settings', title: 'Settings', icon: Settings, defaultRegion: 'left', activityId: 'settings' },
};

export const activityRegistry: ActivityRegistration[] = [
  { id: 'scene', title: 'Scene', icon: FolderTree, panelId: 'hierarchy' },
  { id: 'assets', title: 'Assets', icon: Database, panelId: 'assetExplorer' },
  { id: 'search', title: 'Search', icon: Search, panelId: 'search' },
  { id: 'versionControl', title: 'Version Control', icon: GitBranch, panelId: 'versionControl' },
  { id: 'aiAssistant', title: 'AI Assistant', icon: Bot, panelId: 'aiAssistant' },
  { id: 'profiler', title: 'Profiler', icon: Gauge, panelId: 'profiler' },
  { id: 'renderGraph', title: 'Render Graph', icon: Layers3, panelId: 'renderGraph' },
  { id: 'settings', title: 'Settings', icon: Settings, panelId: 'settings' },
];

export const dockPanelIds = {
  center: ['viewport', 'renderGraph', 'shaderEditor'] satisfies WorkbenchPanelId[],
  right: ['inspector', 'lighting', 'worldSettings'] satisfies WorkbenchPanelId[],
  bottom: ['contentBrowser', 'console', 'versionControl', 'aiAssistant', 'profiler'] satisfies WorkbenchPanelId[],
};

export const getPanel = (id: WorkbenchPanelId) => panelRegistry[id];
