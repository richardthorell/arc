export type StartupState = {
  appVersion: string;
  engineHostConnected: boolean;
  viewportMode: 'unavailable' | 'native' | 'streamed';
  hostError?: string;
};

export type ActivityId =
  'scene' | 'assets' | 'search' | 'versionControl' | 'aiAssistant' | 'profiler' | 'renderGraph' | 'settings';

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
  | 'buildOutput'
  | 'versionControl'
  | 'aiAssistant'
  | 'profiler'
  | 'settings';

export type CommandId =
  | 'file.new'
  | 'file.open'
  | 'file.save'
  | 'file.saveAs'
  | 'file.importScene'
  | 'project.close'
  | 'edit.undo'
  | 'edit.redo'
  | 'entity.duplicate'
  | 'entity.delete'
  | 'scene.play'
  | 'scene.pause'
  | 'scene.stop'
  | 'scene.step'
  | 'scene.buildLighting'
  | 'viewport.select'
  | 'viewport.translate'
  | 'viewport.rotate'
  | 'viewport.scale'
  | 'viewport.terrain'
  | 'viewport.frameSelected'
  | 'layout.reset'
  | 'layout.levelDesign'
  | 'layout.materials'
  | 'layout.profiling'
  | 'view.commandPalette'
  | 'assets.import'
  | 'assets.saveAll'
  | 'vcs.commit'
  | 'vcs.pull'
  | 'vcs.push'
  | 'ai.newChat'
  | 'settings.open';

// Keep registry icons as the concrete Lucide component type. Widening them to
// LucideIcon loses the JSX component signature with the current React typings.
export type WorkbenchIcon = (typeof import('lucide-react'))['FolderTree'];

export type PanelRegistration = {
  id: WorkbenchPanelId;
  title: string;
  icon: WorkbenchIcon | null;
  defaultRegion: DockRegion;
  activityId?: ActivityId;
  allowMultiple?: boolean;
  minimumWidth?: number;
  minimumHeight?: number;
  closeable?: boolean;
};

export type CommandContext = {
  editorFocused: boolean;
  viewportFocused: boolean;
  textInputFocused: boolean;
  modalOpen: boolean;
  playing: boolean;
  hasSelection: boolean;
  canUndo: boolean;
  canRedo: boolean;
  projectOpen: boolean;
};

export type ActivityRegistration = {
  id: ActivityId;
  title: string;
  icon: WorkbenchIcon;
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
