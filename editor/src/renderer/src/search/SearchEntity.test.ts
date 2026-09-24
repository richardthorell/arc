import { describe, expect, it } from 'vitest';

import type { CommandRegistration } from '../app/commandRegistry';
import type { CommandContext } from '../app/workbenchTypes';
import type { AssetItem } from '../services/editorHostTypes';
import { AssetSearchEntity, CommandSearchEntity, SearchEntity } from './SearchEntity';

const asset: AssetItem = {
  id: 'asset-1',
  name: 'T_Bark_Albedo',
  path: 'Assets/Textures/T_Bark_Albedo.png',
  kind: 'texture',
  status: 'ready',
  scope: 'project',
};

const command: CommandRegistration = {
  id: 'edit.undo',
  label: 'Undo',
  description: 'Undo the last scene edit.',
  category: 'Edit',
  defaultKeybindings: ['Ctrl+Z'],
  enabled: (context) => context.canUndo,
  disabledReason: () => 'There is nothing to undo',
};

const context = (canUndo: boolean): CommandContext => ({
  editorFocused: true,
  viewportFocused: false,
  textInputFocused: false,
  modalOpen: false,
  playing: false,
  hasSelection: false,
  projectOpen: true,
  canUndo,
  canRedo: false,
});

describe('SearchEntity', () => {
  it('uses a shared base class for asset and command results', () => {
    expect(new AssetSearchEntity(asset)).toBeInstanceOf(SearchEntity);
    expect(new CommandSearchEntity(command, context(true))).toBeInstanceOf(SearchEntity);
  });

  it('matches asset display data and metadata across multiple terms', () => {
    const result = new AssetSearchEntity(asset);

    expect(result.matches('bark texture')).toBe(true);
    expect(result.matches('project ready')).toBe(true);
    expect(result.matches('material')).toBe(false);
  });

  it('exposes command shortcuts and availability from the current context', () => {
    const enabled = new CommandSearchEntity(command, context(true));
    const disabled = new CommandSearchEntity(command, context(false));

    expect(enabled.shortcut).toBe('Ctrl+Z');
    expect(enabled.disabled).toBe(false);
    expect(disabled.disabled).toBe(true);
    expect(disabled.disabledReason).toBe('There is nothing to undo');
    expect(disabled.matches('undo ctrl+z')).toBe(true);
  });
});
