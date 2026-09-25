type HostResponse<T> = {
  succeeded: boolean;
  payload?: T;
};

type AssetThumbnailSnapshot = {
  dataUrl: string;
};

const cachedPreviewRequests = new Map<string, Promise<string | null>>();

export const cachedAssetPreviewKey = (path: string, generation = 0) =>
  `${path.replaceAll('\\', '/').toLocaleLowerCase()}:${generation}`;

/**
 * Reads an already-generated native asset thumbnail without asking the host to
 * create one. maxSize=0 is the host protocol's cache-only thumbnail lookup.
 */
export function loadCachedAssetPreview(path: string, generation = 0): Promise<string | null> {
  const key = cachedAssetPreviewKey(path, generation);
  const cached = cachedPreviewRequests.get(key);
  if (cached) return cached;

  const request = (async () => {
    if (!path || !window.arc?.host) return null;
    const response = (await window.arc.host.query('asset.thumbnail', {
      path,
      maxSize: 0,
    })) as HostResponse<AssetThumbnailSnapshot>;
    return response.succeeded && response.payload?.dataUrl ? response.payload.dataUrl : null;
  })().catch(() => null);

  cachedPreviewRequests.set(key, request);
  void request.then((preview) => {
    if (!preview && cachedPreviewRequests.get(key) === request) cachedPreviewRequests.delete(key);
  });
  return request;
}
