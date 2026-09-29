export type ShaderDiagnosticLocation = {
  path?: string;
  line?: number;
  column?: number;
};

export type ShaderNavigationTarget = {
  path: string;
  line: number;
  column: number;
};

function normalizePath(path: string): string {
  return path.trim().replaceAll('\\', '/');
}

/**
 * Resolve compiler diagnostic locations into a stable source-navigation target.
 *
 * Diagnostics without a positive line are intentionally non-navigable. Relative
 * compiler paths are preserved instead of being guessed against the active
 * document so callers can decide whether to open an include or the current file.
 */
export function resolveShaderDiagnosticTarget(
  diagnostic: ShaderDiagnosticLocation,
  activePath?: string,
): ShaderNavigationTarget | undefined {
  const line = diagnostic.line;
  if (!Number.isInteger(line) || (line ?? 0) <= 0) return undefined;

  const diagnosticPath = diagnostic.path ? normalizePath(diagnostic.path) : '';
  const fallbackPath = activePath ? normalizePath(activePath) : '';
  const path = diagnosticPath || fallbackPath;
  if (!path) return undefined;

  const rawColumn = diagnostic.column;
  const column = Number.isInteger(rawColumn) && (rawColumn ?? 0) > 0 ? rawColumn! : 1;
  return { path, line: line!, column };
}

export function shaderDiagnosticTargetKey(target: ShaderNavigationTarget): string {
  return `${target.path}:${target.line}:${target.column}`;
}
