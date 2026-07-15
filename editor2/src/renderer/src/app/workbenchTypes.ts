import type { LucideIcon } from 'lucide-react';

export type StartupState = {
  appVersion: string;
  engineHostConnected: boolean;
  viewportMode: 'placeholder' | 'native' | 'streamed';
  hostError?: string;
};

export type ActivityId = 'scene' | 'assets' | 'search' | 'versionControl' | 'aiAssistant' | 'profiler' | 'renderGraph' | 'settings';

export type DockRegion = 'left' | 'center' | 'right' | 'bottom';

export type WorkbenchPanelId =
  | 'hierarchy'
  | 'assetExplorer'
  | 'search'
  | 'viewport'
  | 'renderGraph'
  | 'shaderEditor'
  | 'inspector'
  | 'lighting'
  | 'worldSettings'
  | 'contentBrowser'
  | 'console'
  | 'versionControl'
  | 'aiAssistant'
  | 'profiler'
  | 'settings';

export type CommandId =
  | 'file.open'
  | 'file.importScene'
  | 'scene.play'
  | 'scene.pause'
  | 'scene.stop'
  | 'scene.step'
  | 'scene.buildLighting'
  | 'viewport.select'
  | 'viewport.translate'
  | 'viewport.rotate'
  | 'viewport.scale'
  | 'viewport.frameSelected'
  | 'layout.reset'
  | 'assets.import'
  | 'assets.saveAll'
  | 'vcs.commit'
  | 'vcs.pull'
  | 'vcs.push'
  | 'ai.newChat'
  | 'settings.open';

export type PanelRegistration = {
  id: WorkbenchPanelId;
  title: string;
  icon: LucideIcon;
  defaultRegion: DockRegion;
  activityId?: ActivityId;
};

export type ActivityRegistration = {
  id: ActivityId;
  title: string;
  icon: LucideIcon;
  panelId: WorkbenchPanelId;
};

export type WorkbenchLayoutState = {
  activeActivity: ActivityId;
  activeCenterPanel: WorkbenchPanelId;
  activeRightPanel: WorkbenchPanelId;
  activeBottomPanel: WorkbenchPanelId;
  activityExpanded: boolean;
  leftPanelWidth: number;
  rightPanelWidth: number;
  bottomPanelHeight: number;
  leftVisible: boolean;
  rightVisible: boolean;
  bottomVisible: boolean;
};

export type WorkbenchCommandResult = {
  command: CommandId;
  label: string;
  succeeded: boolean;
  message: string;
};
