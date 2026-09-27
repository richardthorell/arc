import { createHash } from 'node:crypto';

export type AgentEditableAssetKind = 'material' | 'flow';

export type AgentEditableAssetSnapshot = {
  kind: AgentEditableAssetKind;
  path: string;
  revision: string;
  definition: Record<string, unknown>;
};

export type AgentAssetMutationRequest = {
  kind: AgentEditableAssetKind;
  path: string;
  expectedRevision: string;
  definition: Record<string, unknown>;
};

export type AgentAssetMutationResult = {
  contents: string;
  revision: string;
  changed: boolean;
};

const asObject = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

export const assetRevision = (contents: string): string =>
  `sha256:${createHash('sha256').update(contents, 'utf8').digest('hex')}`;

export const parseEditableAgentAsset = (
  kind: AgentEditableAssetKind,
  path: string,
  contents: string,
): AgentEditableAssetSnapshot => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(contents);
  } catch {
    throw new Error(`Asset is not valid JSON: ${path}`);
  }
  const definition = asObject(parsed);
  const graph = asObject(definition.graph);
  if (kind === 'material') {
    if (!path.toLowerCase().endsWith('.arcmat')) throw new Error('Material assets must use the .arcmat extension');
    if (
      definition.version !== 4 ||
      graph.version !== 1 ||
      !Array.isArray(graph.nodes) ||
      !Array.isArray(graph.connections)
    )
      throw new Error('Material definition must contain a version-4 asset and version-1 graph');
  } else {
    if (!path.toLowerCase().endsWith('.arcflow')) throw new Error('Flow assets must use the .arcflow extension');
    if (
      definition.version !== 1 ||
      definition.assetType !== 'flow' ||
      graph.version !== 1 ||
      !Array.isArray(graph.variables) ||
      !Array.isArray(graph.nodes) ||
      !Array.isArray(graph.connections)
    )
      throw new Error('Flow definition must contain a version-1 Flow asset and graph');
  }
  return { kind, path, revision: assetRevision(contents), definition };
};

export const prepareAgentAssetMutation = (
  currentContents: string,
  request: AgentAssetMutationRequest,
): AgentAssetMutationResult => {
  const current = parseEditableAgentAsset(request.kind, request.path, currentContents);
  if (!request.expectedRevision || request.expectedRevision !== current.revision) {
    throw new Error(
      `Asset revision conflict for ${request.path}: expected ${request.expectedRevision || '(missing)'}, current ${current.revision}`,
    );
  }

  const candidate = `${JSON.stringify(request.definition, null, 2)}\n`;
  parseEditableAgentAsset(request.kind, request.path, candidate);
  return {
    contents: candidate,
    revision: assetRevision(candidate),
    changed: candidate !== currentContents,
  };
};
