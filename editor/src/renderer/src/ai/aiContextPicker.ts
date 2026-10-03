import type { AiConversationContextReference } from '../../../common/aiConversationTypes';
import type {
  AiContextJsonValue,
  AiContextRevision,
  AiContextSection,
  AiContextSectionId,
  AiProjectContextSnapshot,
} from '../../../common/aiContextTypes';
import type { AiJsonObject } from '../../../common/aiRuntimeTypes';
import type { AiContextProvider } from './aiProjectContextService';

export type AiContextPickerKind =
  | 'selection'
  | 'scene'
  | 'workspace'
  | 'viewport'
  | 'diagnostics'
  | 'entity'
  | 'asset';

export type AiContextPickerCandidate = {
  id: string;
  kind: AiContextPickerKind;
  label: string;
  detail?: string;
  stableId?: string;
  sectionId: AiContextSectionId;
  data: AiContextJsonValue;
  generation?: number;
};

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const stringValue = (value: unknown): string | undefined => (typeof value === 'string' ? value : undefined);
const numberValue = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;

const jsonRevision = (revision: AiContextRevision): AiJsonObject => ({
  ...(revision.sceneRevision !== undefined ? { sceneRevision: revision.sceneRevision } : {}),
  ...(revision.worldEpoch !== undefined ? { worldEpoch: revision.worldEpoch } : {}),
  ...(revision.frameRevision !== undefined ? { frameRevision: revision.frameRevision } : {}),
  ...(revision.eventSequence !== undefined ? { eventSequence: revision.eventSequence } : {}),
});

const section = (snapshot: AiProjectContextSnapshot, id: AiContextSectionId): AiContextSection | undefined =>
  snapshot.sections.find((candidate) => candidate.id === id && candidate.status === 'ready' && candidate.data !== undefined);

const recordData = (candidate: AiContextSection | undefined): Record<string, AiContextJsonValue> | null =>
  candidate?.data && typeof candidate.data === 'object' && !Array.isArray(candidate.data)
    ? (candidate.data as Record<string, AiContextJsonValue>)
    : null;

const displayNameFromPath = (path: string): string => {
  const normalized = path.replaceAll('\\', '/');
  const leaf = normalized.split('/').filter(Boolean).pop() ?? path;
  return leaf.replace(/\.[^.]+$/u, '') || leaf;
};

const flattenEntities = (value: AiContextJsonValue | undefined): Array<Record<string, AiContextJsonValue>> => {
  if (!Array.isArray(value)) return [];
  const result: Array<Record<string, AiContextJsonValue>> = [];
  const visit = (entry: AiContextJsonValue) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return;
    const record = entry as Record<string, AiContextJsonValue>;
    result.push(record);
    const children = record.children;
    if (Array.isArray(children)) children.forEach(visit);
  };
  value.forEach(visit);
  return result;
};

const quickCandidates = (snapshot: AiProjectContextSnapshot): AiContextPickerCandidate[] => {
  const result: AiContextPickerCandidate[] = [];

  const selection = section(snapshot, 'selection');
  const selectionData = recordData(selection);
  if (selection && selectionData) {
    const selectedGuids = Array.isArray(selectionData.selectedGuids)
      ? selectionData.selectedGuids.filter((value): value is string => typeof value === 'string')
      : [];
    const guid = typeof selectionData.guid === 'string' ? selectionData.guid : selectedGuids.length === 1 ? selectedGuids[0] : undefined;
    result.push({
      id: 'selection:current',
      kind: 'selection',
      label: 'Current selection',
      detail: selectedGuids.length ? `${selectedGuids.length} selected` : 'Selection metadata',
      ...(guid ? { stableId: guid } : {}),
      sectionId: 'selection',
      data: selection.data!,
    });
  }

  const scene = section(snapshot, 'scene');
  const sceneData = recordData(scene);
  if (scene && sceneData) {
    const entities = flattenEntities(sceneData.entities);
    result.push({
      id: `scene:${typeof sceneData.sceneGuid === 'string' ? sceneData.sceneGuid : 'current'}`,
      kind: 'scene',
      label: 'Current scene',
      detail: `${entities.length} ${entities.length === 1 ? 'entity' : 'entities'}`,
      ...(typeof sceneData.sceneGuid === 'string' ? { stableId: sceneData.sceneGuid } : {}),
      sectionId: 'scene',
      data: scene.data!,
    });
  }

  const workspace = section(snapshot, 'workspace');
  const workspaceData = recordData(workspace);
  if (workspace && workspaceData) {
    const active = typeof workspaceData.active === 'string' ? workspaceData.active : undefined;
    result.push({
      id: `workspace:${active ?? 'current'}`,
      kind: 'workspace',
      label: 'Open asset/editor',
      detail: active ?? 'Open editor documents',
      ...(active ? { stableId: active } : {}),
      sectionId: 'workspace',
      data: workspace.data!,
    });
  }

  const viewport = section(snapshot, 'viewport');
  const viewportData = recordData(viewport);
  if (viewport && viewportData) {
    const viewportId = typeof viewportData.viewportId === 'string' ? viewportData.viewportId : 'viewport-1';
    result.push({
      id: `viewport:${viewportId}`,
      kind: 'viewport',
      label: 'Viewport',
      detail: viewportId,
      sectionId: 'viewport',
      data: viewport.data!,
    });
  }

  const diagnostics = section(snapshot, 'diagnostics');
  if (diagnostics) {
    result.push({
      id: 'diagnostics:current',
      kind: 'diagnostics',
      label: 'Diagnostics',
      detail: 'Current editor and host diagnostics',
      sectionId: 'diagnostics',
      data: diagnostics.data!,
    });
  }

  return result;
};

const entityCandidates = (snapshot: AiProjectContextSnapshot): AiContextPickerCandidate[] => {
  const scene = section(snapshot, 'scene');
  const sceneData = recordData(scene);
  if (!scene || !sceneData) return [];
  return flattenEntities(sceneData.entities)
    .flatMap((entity): AiContextPickerCandidate[] => {
      const guid = typeof entity.guid === 'string' ? entity.guid : undefined;
      if (!guid) return [];
      const name = typeof entity.name === 'string' && entity.name.trim() ? entity.name : guid;
      return [
        {
          id: `entity:${guid}`,
          kind: 'entity',
          label: name,
          detail: 'Scene entity',
          stableId: guid,
          sectionId: 'scene',
          data: entity,
        },
      ];
    })
    .sort((left, right) => left.label.localeCompare(right.label));
};

const assetCandidates = (snapshot: AiProjectContextSnapshot): AiContextPickerCandidate[] => {
  const assets = section(snapshot, 'assets');
  const assetsData = recordData(assets);
  const entries = assetsData?.assets;
  if (!assets || !Array.isArray(entries)) return [];
  return entries
    .flatMap((value): AiContextPickerCandidate[] => {
      if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
      const asset = value as Record<string, AiContextJsonValue>;
      const guid = typeof asset.guid === 'string' ? asset.guid : undefined;
      if (!guid) return [];
      const path = typeof asset.path === 'string' ? asset.path : '';
      const label =
        typeof asset.name === 'string' && asset.name.trim() ? asset.name : path ? displayNameFromPath(path) : guid;
      const type = typeof asset.typeId === 'string' ? asset.typeId : 'Asset';
      const generation = typeof asset.generation === 'number' ? asset.generation : undefined;
      return [
        {
          id: `asset:${guid}`,
          kind: 'asset',
          label,
          detail: path ? `${type} · ${path}` : type,
          stableId: guid,
          sectionId: 'assets',
          data: asset,
          ...(generation !== undefined ? { generation } : {}),
        },
      ];
    })
    .sort((left, right) => left.label.localeCompare(right.label));
};

export const collectAiContextPickerCandidates = (snapshot: AiProjectContextSnapshot): AiContextPickerCandidate[] => [
  ...quickCandidates(snapshot),
  ...entityCandidates(snapshot),
  ...assetCandidates(snapshot),
];

export const aiContextReferenceFromCandidate = (
  snapshot: AiProjectContextSnapshot,
  candidate: AiContextPickerCandidate,
): AiConversationContextReference => {
  const candidateSection = snapshot.sections.find((value) => value.id === candidate.sectionId);
  const revision = { ...snapshot.revision, ...(candidateSection?.freshness.revision ?? {}) };
  const metadata: AiJsonObject = {
    ...(snapshot.projectGuid ? { projectGuid: snapshot.projectGuid } : {}),
    capturedAt: candidateSection?.freshness.capturedAt ?? snapshot.capturedAt,
    section: candidate.sectionId,
    revision: jsonRevision(revision),
    data: candidate.data,
    ...(candidate.generation !== undefined ? { assetGeneration: candidate.generation } : {}),
  };
  return {
    id: candidate.id,
    kind: candidate.kind,
    label: candidate.label,
    ...(candidate.stableId ? { stableId: candidate.stableId } : {}),
    metadata,
  };
};

export const captureAiViewportReference = (
  snapshot: AiProjectContextSnapshot,
  root: Pick<Document, 'getElementById'> = document,
): AiConversationContextReference => {
  const viewport = section(snapshot, 'viewport');
  const viewportData = recordData(viewport);
  const viewportId = typeof viewportData?.viewportId === 'string' ? viewportData.viewportId : 'viewport-1';
  const safeViewportId = viewportId.replaceAll(/[^a-zA-Z0-9_-]/g, '-');
  const canvas = root.getElementById(`arc-viewport-surface-${safeViewportId}`);
  if (!(canvas instanceof HTMLCanvasElement) || canvas.width <= 0 || canvas.height <= 0)
    throw new Error('Viewport capture is unavailable until the streamed viewport has rendered a frame');

  return {
    id: `viewport-capture:${snapshot.collectionId}:${Date.now().toString(36)}`,
    kind: 'viewportCapture',
    label: 'Viewport capture',
    metadata: {
      ...(snapshot.projectGuid ? { projectGuid: snapshot.projectGuid } : {}),
      capturedAt: new Date().toISOString(),
      viewportId,
      width: canvas.width,
      height: canvas.height,
    },
    attachment: {
      type: 'image',
      mimeType: 'image/png',
      uri: canvas.toDataURL('image/png'),
      alt: `ARC ${viewportId} capture`,
    },
  };
};

export const createAiAssetContextProvider = (): AiContextProvider => ({
  id: 'assets',
  async collect({ environment, projectGuid }) {
    if (!projectGuid) return { status: 'unavailable' };
    const raw = await environment.hostQuery('project.assets', {});
    const response = asRecord(raw);
    if (!response) return { status: 'unavailable' };
    if (response.succeeded === false)
      return {
        status: 'error',
        error: stringValue(response.error) ?? 'project.assets query failed',
      };
    const payload = asRecord(response.payload);
    const assets = Array.isArray(payload?.assets)
      ? payload.assets.flatMap((value) => {
          const asset = asRecord(value);
          const guid = stringValue(asset?.guid);
          if (!asset || !guid) return [];
          return [
            {
              guid,
              path: stringValue(asset.path) ?? '',
              name: stringValue(asset.name) ?? '',
              typeId: stringValue(asset.typeId) ?? '',
              state: stringValue(asset.state) ?? '',
              ...(numberValue(asset.generation) !== undefined ? { generation: numberValue(asset.generation) } : {}),
            },
          ];
        })
      : [];
    const revision: AiContextRevision = {
      sceneRevision: numberValue(response.sceneRevision),
      worldEpoch: numberValue(response.worldEpoch),
      frameRevision: numberValue(response.frameRevision),
    };
    return {
      status: 'ready',
      data: { assets },
      ...(Object.values(revision).some((value) => value !== undefined) ? { revision } : {}),
    };
  },
});
