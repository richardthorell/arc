export type ShaderIncludeTarget = {
  path: string;
};

function normalizePath(path: string): string {
  return path.trim().replaceAll('\\', '/');
}

function collapseSegments(path: string): string {
  const absolute = path.startsWith('/');
  const segments: string[] = [];
  for (const segment of path.split('/')) {
    if (!segment || segment === '.') continue;
    if (segment === '..') {
      if (segments.length && segments.at(-1) !== '..') segments.pop();
      else if (!absolute) segments.push(segment);
      continue;
    }
    segments.push(segment);
  }
  return `${absolute ? '/' : ''}${segments.join('/')}`;
}

/**
 * Resolve an authored shader include into a stable navigation target without
 * depending on editor UI state. Relative includes are resolved from the active
 * shader's directory; project/root-style paths remain unchanged.
 */
export function resolveShaderIncludeTarget(
  includePath: string,
  activePath?: string,
): ShaderIncludeTarget | undefined {
  const include = normalizePath(includePath);
  if (!include) return undefined;

  if (include.startsWith('/') || /^[A-Za-z]:\//.test(include)) {
    return { path: collapseSegments(include) };
  }

  const active = activePath ? normalizePath(activePath) : '';
  if (!active) return { path: collapseSegments(include) };

  const separator = active.lastIndexOf('/');
  const directory = separator >= 0 ? active.slice(0, separator) : '';
  return { path: collapseSegments(directory ? `${directory}/${include}` : include) };
}
