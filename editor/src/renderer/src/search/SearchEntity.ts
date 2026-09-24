import type { CommandRegistration } from '../app/commandRegistry';
import type { CommandContext } from '../app/workbenchTypes';
import type { AssetItem } from '../services/editorHostTypes';

export type SearchEntityVariant = 'asset' | 'command';

export type SearchEntityOptions = {
  id: string;
  title: string;
  subtitle?: string;
  meta?: string;
  state?: string;
  shortcut?: string;
  disabled?: boolean;
  disabledReason?: string;
  keywords?: readonly string[];
};

export abstract class SearchEntity {
  abstract readonly variant: SearchEntityVariant;

  readonly id: string;
  readonly title: string;
  readonly subtitle?: string;
  readonly meta?: string;
  readonly state?: string;
  readonly shortcut?: string;
  readonly disabled: boolean;
  readonly disabledReason?: string;

  private readonly searchText: string;

  protected constructor(options: SearchEntityOptions) {
    this.id = options.id;
    this.title = options.title;
    this.subtitle = options.subtitle;
    this.meta = options.meta;
    this.state = options.state;
    this.shortcut = options.shortcut;
    this.disabled = options.disabled ?? false;
    this.disabledReason = options.disabledReason;
    this.searchText = [
      options.title,
      options.subtitle,
      options.meta,
      options.state,
      options.shortcut,
      ...(options.keywords ?? []),
    ]
      .filter(Boolean)
      .join(' ')
      .toLocaleLowerCase();
  }

  matches(query: string): boolean {
    const terms = query
      .trim()
      .toLocaleLowerCase()
      .split(/\s+/)
      .filter(Boolean);
    return terms.every((term) => this.searchText.includes(term));
  }
}

export class AssetSearchEntity extends SearchEntity {
  readonly variant = 'asset' as const;
  readonly asset: AssetItem;

  constructor(asset: AssetItem) {
    super({
      id: `asset:${asset.id}`,
      title: asset.title?.trim() || asset.name,
      subtitle: asset.path,
      meta: asset.kind,
      state: asset.status,
      keywords: [asset.name, asset.scope ?? '', asset.typeId ?? '', asset.importerId ?? '', asset.description ?? ''],
    });
    this.asset = asset;
  }
}

export class CommandSearchEntity extends SearchEntity {
  readonly variant = 'command' as const;
  readonly command: CommandRegistration;

  constructor(command: CommandRegistration, context?: CommandContext) {
    const enabled = !context || !command.enabled || command.enabled(context);
    super({
      id: `command:${command.id}`,
      title: command.label,
      subtitle: command.description,
      meta: command.category,
      shortcut: command.defaultKeybindings?.[0],
      disabled: !enabled,
      disabledReason: enabled || !context ? undefined : command.disabledReason?.(context),
      keywords: [command.id, ...(command.defaultKeybindings ?? [])],
    });
    this.command = command;
  }
}
