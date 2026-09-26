import type {
  ArcAssetDownloadFile,
  ArcAssetDownloadManifest,
  ArcRemoteAssetKind,
} from '../../../common/assetSourceTypes';

const knownFormats = new Set(['hdr', 'exr', 'jpg', 'jpeg', 'png', 'blend', 'gltf', 'glb', 'fbx', 'usd', 'usdz']);
const formatFamilies: Partial<Record<ArcRemoteAssetKind, ReadonlySet<string>>> = {
  hdri: new Set(['hdr', 'exr']),
  model: new Set(['blend', 'gltf', 'glb', 'fbx', 'usd', 'usdz']),
  texture: new Set(['jpg', 'jpeg', 'png', 'exr']),
};

const normalizeVariantToken = (value: string): string => value.trim().replace(/^\./, '').toLocaleLowerCase();

const segments = (file: ArcAssetDownloadFile): string[] =>
  file.logicalPath
    .replaceAll('\\', '/')
    .split('/')
    .filter(Boolean)
    .map(normalizeVariantToken);

const extension = (file: ArcAssetDownloadFile): string => {
  try {
    return normalizeVariantToken(new URL(file.url).pathname.split('.').at(-1) ?? '');
  } catch {
    return '';
  }
};

const resolutionValue = (value: string): number => {
  const match = value.match(/^(\d+(?:\.\d+)?)k$/i);
  return match ? Number.parseFloat(match[1]) : Number.POSITIVE_INFINITY;
};

export const manifestResolutions = (manifest: ArcAssetDownloadManifest): string[] =>
  Array.from(
    new Set(manifest.files.flatMap((file) => segments(file).filter((segment) => /^\d+(?:\.\d+)?k$/i.test(segment)))),
  ).sort((left, right) => resolutionValue(left) - resolutionValue(right));

export const manifestFormats = (manifest: ArcAssetDownloadManifest, kind?: ArcRemoteAssetKind): string[] => {
  const allowed = kind ? (formatFamilies[kind] ?? knownFormats) : knownFormats;
  return Array.from(
    new Set(
      manifest.files.flatMap((file) => {
        const candidates = [...segments(file).filter((segment) => knownFormats.has(segment)), extension(file)];
        return candidates.filter((candidate) => allowed.has(candidate));
      }),
    ),
  ).sort();
};

export const preferredResolution = (resolutions: string[]): string => {
  const normalized = resolutions.map(normalizeVariantToken);
  const twoK = normalized.indexOf('2k');
  if (twoK >= 0) return resolutions[twoK];
  const fourK = normalized.indexOf('4k');
  if (fourK >= 0) return resolutions[fourK];
  return resolutions[0] ?? '';
};

export const preferredFormat = (formats: string[], kind: ArcRemoteAssetKind): string => {
  const preferences =
    kind === 'hdri'
      ? ['hdr', 'exr']
      : kind === 'model'
        ? ['gltf', 'glb', 'fbx', 'usd', 'usdz', 'blend']
        : ['jpg', 'png', 'exr'];
  const normalized = formats.map(normalizeVariantToken);
  for (const preference of preferences) {
    const index = normalized.indexOf(preference);
    if (index >= 0) return formats[index];
  }
  return formats[0] ?? '';
};

export const selectManifestFiles = (
  manifest: ArcAssetDownloadManifest,
  resolution: string,
  format: string,
): ArcAssetDownloadFile[] => {
  const normalizedResolution = normalizeVariantToken(resolution);
  const normalizedFormat = normalizeVariantToken(format);
  return manifest.files.filter((file) => {
    const fileSegments = segments(file);
    const resolutionMatches = !normalizedResolution || fileSegments.includes(normalizedResolution);
    const formatMatches =
      !normalizedFormat || fileSegments.includes(normalizedFormat) || extension(file) === normalizedFormat;
    return resolutionMatches && formatMatches;
  });
};

export const manifestSelectionBytes = (files: ArcAssetDownloadFile[]): number | undefined =>
  files.every((file) => file.sizeBytes !== undefined)
    ? files.reduce((sum, file) => sum + (file.sizeBytes ?? 0), 0)
    : undefined;
