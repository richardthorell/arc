import type { CommandContext, CommandId } from './workbenchTypes';

export type CommandRegistration = {
  id: CommandId;
  label: string;
  description: string;
  category: 'File' | 'Edit' | 'Entity' | 'Scene' | 'Viewport' | 'Layout' | 'Assets' | 'Tools';
  defaultKeybindings?: readonly string[];
  enabled?: (context: CommandContext) => boolean;
  disabledReason?: (context: CommandContext) => string;
};

export const commandRegistry: Record<CommandId, CommandRegistration> = {
  'file.new': {
    id: 'file.new',
    label: 'New Scene',
    description: 'Create a new untitled scene.',
    category: 'File',
    defaultKeybindings: ['Ctrl+N'],
    enabled: (context) => context.projectOpen,
  },
  'file.open': {
    id: 'file.open',
    label: 'Open Scene',
    description: 'Open a scene asset and replace imported scene content.',
    category: 'File',
    defaultKeybindings: ['Ctrl+O'],
    enabled: (context) => context.projectOpen,
  },
  'file.save': {
    id: 'file.save',
    label: 'Save Scene',
    description: 'Save the active scene.',
    category: 'File',
    defaultKeybindings: ['Ctrl+S'],
    enabled: (context) => context.projectOpen,
  },
  'file.saveAs': {
    id: 'file.saveAs',
    label: 'Save Scene As',
    description: 'Save the active scene to a new path.',
    category: 'File',
    defaultKeybindings: ['Ctrl+Shift+S'],
    enabled: (context) => context.projectOpen,
  },
  'file.importScene': {
    id: 'file.importScene',
    label: 'Import Scene Into Current',
    description: 'Append a scene asset to the current scene.',
    category: 'File',
    enabled: (context) => context.projectOpen,
  },
  'project.close': {
    id: 'project.close',
    label: 'Close Project',
    description: 'Close the active project and return to the project browser.',
    category: 'File',
    enabled: (context) => context.projectOpen,
  },
  'edit.undo': {
    id: 'edit.undo',
    label: 'Undo',
    description: 'Undo the last scene edit.',
    category: 'Edit',
    defaultKeybindings: ['Ctrl+Z'],
    enabled: (context) => context.canUndo,
    disabledReason: () => 'There is nothing to undo',
  },
  'edit.redo': {
    id: 'edit.redo',
    label: 'Redo',
    description: 'Redo the last undone scene edit.',
    category: 'Edit',
    defaultKeybindings: ['Ctrl+Y', 'Ctrl+Shift+Z'],
    enabled: (context) => context.canRedo,
    disabledReason: () => 'There is nothing to redo',
  },
  'entity.duplicate': {
    id: 'entity.duplicate',
    label: 'Duplicate Entity',
    description: 'Duplicate the selected entity subtree.',
    category: 'Entity',
    defaultKeybindings: ['Ctrl+D'],
    enabled: (context) => context.hasSelection,
    disabledReason: () => 'Select an entity first',
  },
  'entity.delete': {
    id: 'entity.delete',
    label: 'Delete Entity',
    description: 'Delete the selected entity subtree.',
    category: 'Entity',
    defaultKeybindings: ['Delete'],
    enabled: (context) => context.hasSelection,
    disabledReason: () => 'Select an entity first',
  },
  'scene.play': { id: 'scene.play', label: 'Play', description: 'Start scene play mode.', category: 'Scene' },
  'scene.pause': {
    id: 'scene.pause',
    label: 'Pause',
    description: 'Pause scene play mode.',
    category: 'Scene',
    enabled: (context) => context.playing,
  },
  'scene.stop': {
    id: 'scene.stop',
    label: 'Stop',
    description: 'Stop scene play mode.',
    category: 'Scene',
    enabled: (context) => context.playing,
  },
  'scene.step': { id: 'scene.step', label: 'Step', description: 'Step one frame.', category: 'Scene' },
  'scene.buildLighting': {
    id: 'scene.buildLighting',
    label: 'Build Lighting',
    description: 'Queue a lighting build.',
    category: 'Scene',
  },
  'viewport.select': {
    id: 'viewport.select',
    label: 'Select Tool',
    description: 'Activate select tool.',
    category: 'Viewport',
    defaultKeybindings: ['Q'],
    enabled: (context) => context.viewportFocused,
  },
  'viewport.translate': {
    id: 'viewport.translate',
    label: 'Translate Tool',
    description: 'Activate translate gizmo.',
    category: 'Viewport',
    defaultKeybindings: ['W'],
    enabled: (context) => context.viewportFocused,
  },
  'viewport.rotate': {
    id: 'viewport.rotate',
    label: 'Rotate Tool',
    description: 'Activate rotate gizmo.',
    category: 'Viewport',
    defaultKeybindings: ['E'],
    enabled: (context) => context.viewportFocused,
  },
  'viewport.scale': {
    id: 'viewport.scale',
    label: 'Scale Tool',
    description: 'Activate scale gizmo.',
    category: 'Viewport',
    defaultKeybindings: ['R'],
    enabled: (context) => context.viewportFocused,
  },
  'viewport.terrain': {
    id: 'viewport.terrain',
    label: 'Terrain Tool',
    description: 'Sculpt or paint the selected terrain.',
    category: 'Viewport',
    // Clicking the main toolbar necessarily moves DOM focus out of the viewport.
    // Terrain eligibility is validated from the selected entity snapshot and by
    // the native host, so requiring transient viewport focus makes the button
    // reject the very click intended to activate it.
    enabled: (context) => context.hasSelection,
  },
  'viewport.frameSelected': {
    id: 'viewport.frameSelected',
    label: 'Frame Selected',
    description: 'Frame the selected object in the viewport.',
    category: 'Viewport',
    defaultKeybindings: ['F'],
    enabled: (context) => context.viewportFocused && context.hasSelection,
  },
  'viewport.snapToFloor': {
    id: 'viewport.snapToFloor',
    label: 'Snap to Floor',
    description: 'Drop the selected entity onto the nearest queryable surface below it.',
    category: 'Viewport',
    defaultKeybindings: ['End'],
    enabled: (context) => context.viewportFocused && context.hasSelection,
    disabledReason: () => 'Focus the viewport and select an entity first',
  },
  'layout.reset': {
    id: 'layout.reset',
    label: 'Reset Layout',
    description: 'Reset the current workbench layout.',
    category: 'Layout',
  },
  'layout.levelDesign': {
    id: 'layout.levelDesign',
    label: 'Level Design Layout',
    description: 'Switch to the Level Design workspace.',
    category: 'Layout',
  },
  'layout.materials': {
    id: 'layout.materials',
    label: 'Materials Layout',
    description: 'Switch to the Materials workspace.',
    category: 'Layout',
  },
  'layout.profiling': {
    id: 'layout.profiling',
    label: 'Profiling Layout',
    description: 'Switch to the Profiling workspace.',
    category: 'Layout',
  },
  'view.commandPalette': {
    id: 'view.commandPalette',
    label: 'Show Command Palette',
    description: 'Search and run editor commands.',
    category: 'Tools',
    defaultKeybindings: ['Ctrl+Shift+P', 'Ctrl+K'],
  },
  'assets.import': {
    id: 'assets.import',
    label: 'Import Asset',
    description: 'Import assets into the project.',
    category: 'Assets',
    enabled: (context) => context.projectOpen,
  },
  'assets.saveAll': {
    id: 'assets.saveAll',
    label: 'Save All',
    description: 'Save all dirty assets and scenes.',
    category: 'File',
    defaultKeybindings: ['Ctrl+Shift+Alt+S'],
    enabled: (context) => context.projectOpen,
  },
  'vcs.commit': { id: 'vcs.commit', label: 'Commit', description: 'Commit staged changes.', category: 'Tools' },
  'vcs.pull': { id: 'vcs.pull', label: 'Pull', description: 'Pull from remote.', category: 'Tools' },
  'vcs.push': { id: 'vcs.push', label: 'Push', description: 'Push to remote.', category: 'Tools' },
  'ai.newChat': {
    id: 'ai.newChat',
    label: 'New AI Chat',
    description: 'Start a new assistant chat.',
    category: 'Tools',
  },
  'settings.open': {
    id: 'settings.open',
    label: 'Open Settings',
    description: 'Open editor settings.',
    category: 'Tools',
    defaultKeybindings: ['Ctrl+,'],
  },
};

export const allCommands = Object.values(commandRegistry);
