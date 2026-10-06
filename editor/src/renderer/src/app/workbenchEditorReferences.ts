import { flattenScene, type AssetItem, type ProjectSnapshot, type SceneEntity } from '../services/editorHostTypes';
import type { ArcResourceRegistry } from '../services/arcResourceRegistry';
import { createWindowArcResourceRegistry } from '../services/arcResources';
import { arcUri } from '../services/arcUri';
import {
  createEditorReferenceController,
  type EditorReferenceController,
  type ResolvedEditorReference,
} from '../services/editorReferences';

export type WorkbenchEditorReferenceActions = {
  getProject: () => ProjectSnapshot | null;
  getActiveScene?: () => { guid: string; name: string } | null;
  selectEntity: (entityId: string) => void | Promise<void>;
  focusSelectedEntity: () => void | Promise<void>;
  selectAsset: (asset: AssetItem) => void | Promise<void>;
  openAsset: (asset: AssetItem) => void | Promise<void>;
  selectScene?: (sceneGuid: string) => void | Promise<void>;
  highlightEntity?: (entity: SceneEntity, active: boolean) => void | Promise<void>;
  highlightAsset?: (asset: AssetItem, active: boolean) => void | Promise<void>;
  resources?: Pick<ArcResourceRegistry, 'read'>;
};

const titleCase = (value: string) =>
  value.replace(/(^|[-_])([a-z])/gu, (_, prefix, letter) => `${prefix ? ' ' : ''}${letter.toUpperCase()}`);

export const findReferencedEntity = (project: ProjectSnapshot | null, guid: string): SceneEntity | null =>
  project ? (flattenScene(project.scene).find((entity) => entity.guid === guid) ?? null) : null;

export const findReferencedAsset = (project: ProjectSnapshot | null, guid: string): AssetItem | null =>
  project ? (project.assets.find((asset) => asset.guid === guid || asset.id === guid) ?? null) : null;

const resolvedEntity = (entity: SceneEntity): Omit<ResolvedEditorReference, 'reference'> => ({
  label: entity.name,
  subtitle: titleCase(entity.kind),
});

const resolvedAsset = (asset: AssetItem): Omit<ResolvedEditorReference, 'reference'> => ({
  label: asset.title?.trim() || asset.name,
  subtitle: titleCase(asset.kind),
});

export const createWorkbenchEditorReferenceController = (
  actions: WorkbenchEditorReferenceActions,
): EditorReferenceController => {
  const entity = (guid: string) => findReferencedEntity(actions.getProject(), guid);
  const asset = (guid: string) => findReferencedAsset(actions.getProject(), guid);
  const resources =
    actions.resources ?? (typeof window !== 'undefined' ? createWindowArcResourceRegistry() : undefined);

  return createEditorReferenceController({
    resolveEntity: (guid) => {
      const value = entity(guid);
      return value ? resolvedEntity(value) : null;
    },
    resolveAsset: async (guid) => {
      const value = asset(guid);
      if (!value) return null;
      const resolved = resolvedAsset(value);
      if (!resources) return resolved;
      const thumbnail = await resources.read(
        arcUri({ kind: 'asset', id: guid, path: ['thumbnail'], query: { size: '64' } }),
      );
      return thumbnail?.dataUrl ? { ...resolved, thumbnailUrl: thumbnail.dataUrl } : resolved;
    },
    resolveScene: (guid) => {
      const scene = actions.getActiveScene?.();
      return scene?.guid === guid ? { label: scene.name, subtitle: 'Scene' } : null;
    },
    activateEntity: async (guid) => {
      const value = entity(guid);
      if (value) await actions.selectEntity(value.id);
    },
    focusEntity: async (guid) => {
      const value = entity(guid);
      if (!value) return;
      await actions.selectEntity(value.id);
      await actions.focusSelectedEntity();
    },
    highlightEntity: actions.highlightEntity
      ? async (guid, active) => {
          const value = entity(guid);
          if (value) await actions.highlightEntity?.(value, active);
        }
      : undefined,
    activateAsset: async (guid) => {
      const value = asset(guid);
      if (value) await actions.selectAsset(value);
    },
    focusAsset: async (guid) => {
      const value = asset(guid);
      if (!value) return;
      await actions.selectAsset(value);
      await actions.openAsset(value);
    },
    highlightAsset: actions.highlightAsset
      ? async (guid, active) => {
          const value = asset(guid);
          if (value) await actions.highlightAsset?.(value, active);
        }
      : undefined,
    activateScene: actions.selectScene,
    focusScene: actions.selectScene,
  });
};
