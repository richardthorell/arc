export type ArcUri = {
  kind: string;
  id: string;
  path: string[];
  query: ReadonlyMap<string, string>;
};

const resourceKindPattern = /^[a-z][a-z0-9-]*$/u;

const decodeSegment = (value: string): string | null => {
  try {
    const decoded = decodeURIComponent(value);
    return decoded.trim() ? decoded : null;
  } catch {
    return null;
  }
};

export const parseArcUri = (value: string): ArcUri | null => {
  const trimmed = value.trim();
  if (!trimmed.startsWith('arc://')) return null;

  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return null;
  }

  if (parsed.protocol !== 'arc:' || !resourceKindPattern.test(parsed.hostname)) return null;
  const segments = parsed.pathname
    .split('/')
    .filter(Boolean)
    .map(decodeSegment);
  if (!segments.length || segments.some((segment) => segment === null)) return null;

  const [id, ...path] = segments as string[];
  const query = new Map<string, string>();
  for (const [key, queryValue] of parsed.searchParams.entries()) {
    if (!key || query.has(key)) return null;
    query.set(key, queryValue);
  }

  return { kind: parsed.hostname, id, path, query };
};

export const arcUri = (
  resource: Pick<ArcUri, 'kind' | 'id'> & { path?: readonly string[]; query?: ReadonlyMap<string, string> | Record<string, string> },
): string => {
  if (!resourceKindPattern.test(resource.kind) || !resource.id.trim())
    throw new Error('ARC URI requires a valid resource kind and stable resource ID');

  const path = [resource.id, ...(resource.path ?? [])].map((segment) => {
    if (!segment.trim()) throw new Error('ARC URI path segments must be non-empty');
    return encodeURIComponent(segment);
  });
  const params = new URLSearchParams();
  const entries = resource.query instanceof Map ? [...resource.query.entries()] : Object.entries(resource.query ?? {});
  for (const [key, value] of entries.sort(([left], [right]) => left.localeCompare(right))) {
    if (!key) throw new Error('ARC URI query keys must be non-empty');
    params.set(key, value);
  }
  const query = params.toString();
  return `arc://${resource.kind}/${path.join('/')}${query ? `?${query}` : ''}`;
};
