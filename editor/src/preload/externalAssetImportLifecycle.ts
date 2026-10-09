export type ExternalImportedAssetState = 'unknown' | 'queued' | 'importing' | 'ready' | 'stale' | 'failed';

export type ExternalImportedAsset = {
  guid: string;
  path: string;
  state: ExternalImportedAssetState;
  diagnostic?: string;
};

export type ExternalAssetImportLifecycleBridge = {
  queryAssets: () => Promise<readonly ExternalImportedAsset[]>;
  reimportAsset: (guid: string) => Promise<{ succeeded: boolean; error?: string }>;
};

export type ExternalAssetImportLifecycleOptions = {
  timeoutMs?: number;
  pollMs?: number;
  sleep?: (milliseconds: number) => Promise<void>;
};

const normalizedAssetPath = (value: string) => value.replaceAll('\\', '/').toLocaleLowerCase();

const defaultSleep = (milliseconds: number) => new Promise<void>((resolve) => setTimeout(resolve, milliseconds));

const failedImportError = (asset: ExternalImportedAsset, label: string) =>
  new Error(asset.diagnostic || `${label} import failed: ${asset.path}`);

export const ensureExternalAssetImported = async (
  relativePath: string,
  label: string,
  bridge: ExternalAssetImportLifecycleBridge,
  options: ExternalAssetImportLifecycleOptions = {},
): Promise<ExternalImportedAsset> => {
  const timeoutMs = options.timeoutMs ?? 15_000;
  const pollMs = options.pollMs ?? 75;
  const sleep = options.sleep ?? defaultSleep;
  const deadline = Date.now() + timeoutMs;
  const normalized = normalizedAssetPath(relativePath);
  let reimportRequested = false;

  while (Date.now() < deadline) {
    const assets = await bridge.queryAssets();
    const asset = assets.find((candidate) => normalizedAssetPath(candidate.path) === normalized);
    if (asset) {
      if (asset.state === 'failed') throw failedImportError(asset, label);
      if (asset.state === 'ready') return asset;
      if (!reimportRequested && asset.guid) {
        const response = await bridge.reimportAsset(asset.guid);
        if (!response.succeeded) throw new Error(response.error || `Could not import ${label}: ${relativePath}`);
        reimportRequested = true;
      }
    }
    await sleep(pollMs);
  }

  throw new Error(`Timed out waiting for ${label} import: ${relativePath}`);
};
