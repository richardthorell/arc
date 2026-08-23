export const arcAssetDragMime = 'application/x-arc-asset';
export const arcEnvironmentDragMime = 'application/x-arc-environment';

export type ArcAssetDragPayload = {
  guid: string;
  type: string;
  pathHint: string;
};

type DragDataSource = Pick<DataTransfer, 'getData'>;

const nonEmptyString = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;

/**
 * Decode the stable Content Browser drag payload.
 *
 * Older editor builds used a plain asset path for the same MIME type, so keep
 * accepting that form while the JSON payload is the canonical representation.
 */
export const parseArcAssetDragPayload = (raw: string): ArcAssetDragPayload | null => {
  const value = raw.trim();
  if (!value) return null;

  try {
    const parsed = JSON.parse(value) as Partial<ArcAssetDragPayload> | null;
    if (!parsed || typeof parsed !== 'object' || !nonEmptyString(parsed.pathHint)) return null;
    return {
      guid: typeof parsed.guid === 'string' ? parsed.guid : '',
      type: typeof parsed.type === 'string' ? parsed.type : '',
      pathHint: parsed.pathHint,
    };
  } catch {
    // Backward compatibility with the original plain-path drag payload.
    return { guid: '', type: '', pathHint: value };
  }
};

export const readArcAssetDragPayload = (dataTransfer: DragDataSource): ArcAssetDragPayload | null => {
  const asset = parseArcAssetDragPayload(dataTransfer.getData(arcAssetDragMime));
  if (asset) return asset;

  const environmentPath = dataTransfer.getData(arcEnvironmentDragMime).trim();
  return environmentPath ? { guid: '', type: 'environment', pathHint: environmentPath } : null;
};
