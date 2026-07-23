import type { CommandId, WorkbenchCommandResult } from './workbenchTypes';

export type CommandRegistration = {
  id: CommandId;
  label: string;
  description: string;
};

export const commandRegistry: Record<CommandId, CommandRegistration> = {
  'file.new': { id: 'file.new', label: 'New Scene', description: 'Create a new untitled mountain scene.' },
  'file.open': { id: 'file.open', label: 'Open Scene', description: 'Open a scene asset and replace imported scene content.' },
  'file.save': { id: 'file.save', label: 'Save Scene', description: 'Save the active scene.' },
  'file.saveAs': { id: 'file.saveAs', label: 'Save Scene As', description: 'Save the active scene to a new path.' },
  'file.importScene': { id: 'file.importScene', label: 'Import Scene Into Current', description: 'Append a scene asset to the current scene.' },
  'edit.undo': { id: 'edit.undo', label: 'Undo', description: 'Undo the last scene edit.' },
  'edit.redo': { id: 'edit.redo', label: 'Redo', description: 'Redo the last undone scene edit.' },
  'entity.duplicate': { id: 'entity.duplicate', label: 'Duplicate Entity', description: 'Duplicate the selected entity subtree.' },
  'entity.delete': { id: 'entity.delete', label: 'Delete Entity', description: 'Delete the selected entity subtree.' },
  'scene.play': { id: 'scene.play', label: 'Play', description: 'Start scene play mode.' },
  'scene.pause': { id: 'scene.pause', label: 'Pause', description: 'Pause scene play mode.' },
  'scene.stop': { id: 'scene.stop', label: 'Stop', description: 'Stop scene play mode.' },
  'scene.step': { id: 'scene.step', label: 'Step', description: 'Step one frame.' },
  'scene.buildLighting': { id: 'scene.buildLighting', label: 'Build Lighting', description: 'Queue a lighting build.' },
  'viewport.select': { id: 'viewport.select', label: 'Select Tool', description: 'Activate select tool.' },
  'viewport.translate': { id: 'viewport.translate', label: 'Translate Tool', description: 'Activate translate gizmo.' },
  'viewport.rotate': { id: 'viewport.rotate', label: 'Rotate Tool', description: 'Activate rotate gizmo.' },
  'viewport.scale': { id: 'viewport.scale', label: 'Scale Tool', description: 'Activate scale gizmo.' },
  'viewport.terrain': { id: 'viewport.terrain', label: 'Terrain Tool', description: 'Sculpt or paint the selected terrain.' },
  'viewport.frameSelected': { id: 'viewport.frameSelected', label: 'Frame Selected', description: 'Frame the selected object in the viewport.' },
  'layout.reset': { id: 'layout.reset', label: 'Reset Layout', description: 'Reset the workbench layout.' },
  'assets.import': { id: 'assets.import', label: 'Import Asset', description: 'Import assets into the project.' },
  'assets.saveAll': { id: 'assets.saveAll', label: 'Save All', description: 'Save all dirty assets and scenes.' },
  'vcs.commit': { id: 'vcs.commit', label: 'Commit', description: 'Commit staged changes.' },
  'vcs.pull': { id: 'vcs.pull', label: 'Pull', description: 'Pull from remote.' },
  'vcs.push': { id: 'vcs.push', label: 'Push', description: 'Push to remote.' },
  'ai.newChat': { id: 'ai.newChat', label: 'New AI Chat', description: 'Start a new assistant chat.' },
  'settings.open': { id: 'settings.open', label: 'Open Settings', description: 'Open editor settings.' },
};

export const executeWorkbenchCommand = async (command: CommandId): Promise<WorkbenchCommandResult> => {
  const registration = commandRegistry[command];
  await new Promise((resolve) => window.setTimeout(resolve, 40));

  return {
    command,
    label: registration.label,
    succeeded: true,
    message: `${registration.label} queued`,
  };
};
