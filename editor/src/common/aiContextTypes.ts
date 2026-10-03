export const AI_PROJECT_CONTEXT_SCHEMA_VERSION = 1 as const;

export type AiContextSectionId =
  | 'project'
  | 'scene'
  | 'selection'
  | 'workspace'
  | 'assets'
  | 'diagnostics'
  | 'viewport'
  | 'recentChanges';

export type AiContextJsonValue =
  null | boolean | number | string | AiContextJsonValue[] | { [key: string]: AiContextJsonValue };

export type AiContextRevision = {
  sceneRevision?: number;
  worldEpoch?: number;
  frameRevision?: number;
  eventSequence?: number;
};

export type AiContextCostEstimate = {
  characters: number;
  approximateTokens: number;
};

export type AiContextFreshness = {
  capturedAt: string;
  ageMs: number;
  cache: 'live' | 'cached';
  revision?: AiContextRevision;
};

export type AiContextSection = {
  id: AiContextSectionId;
  status: 'ready' | 'unavailable' | 'error';
  data?: AiContextJsonValue;
  error?: string;
  truncated: boolean;
  freshness: AiContextFreshness;
  estimatedCost: AiContextCostEstimate;
};

export type AiProjectContextSnapshot = {
  schemaVersion: typeof AI_PROJECT_CONTEXT_SCHEMA_VERSION;
  collectionId: string;
  projectGuid: string | null;
  capturedAt: string;
  revision: AiContextRevision;
  sections: AiContextSection[];
  estimatedCost: AiContextCostEstimate;
};

export type AiContextCollectionEvent =
  | {
      type: 'collection.started';
      collectionId: string;
      projectGuid: string | null;
      timestamp: string;
    }
  | {
      type: 'provider.completed';
      collectionId: string;
      providerId: AiContextSectionId;
      status: AiContextSection['status'];
      durationMs: number;
      cache: AiContextFreshness['cache'];
      timestamp: string;
    }
  | {
      type: 'collection.completed';
      collectionId: string;
      projectGuid: string | null;
      estimatedCost: AiContextCostEstimate;
      durationMs: number;
      timestamp: string;
    }
  | {
      type: 'context.invalidated';
      providerIds: AiContextSectionId[] | null;
      reason: string;
      timestamp: string;
    };