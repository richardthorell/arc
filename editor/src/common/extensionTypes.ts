export type ArcExtensionCapability =
  | 'filesystem.read'
  | 'filesystem.write'
  | 'sourceControl'
  | 'asset.read'
  | 'asset.mutate'
  | 'scene.read'
  | 'scene.mutate';

export type ArcExtensionManifest = {
  format: 'arc-extension';
  formatVersion: 1;
  id: string;
  name: string;
  version: string;
  engineVersion: string;
  main: string;
  activationEvents: string[];
  capabilities: ArcExtensionCapability[];
  contributes?: {
    commands?: Array<{ id: string; title: string }>;
    panels?: Array<{ id: string; title: string; entry: string }>;
    propertyDrawers?: Array<{ fieldType: string; entry: string }>;
    assetEditors?: Array<{ assetType: string; entry: string }>;
  };
};

export type ArcExtensionSnapshot = {
  revision: number;
  extensions: Array<{
    manifest: ArcExtensionManifest;
    root: string;
    compatible: boolean;
    enabled: boolean;
    grantedCapabilities: ArcExtensionCapability[];
    diagnostics: string[];
    active?: boolean;
    registeredCommands?: string[];
  }>;
};
