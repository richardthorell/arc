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

export type ShaderDiagnosticRoute =
  | { kind: 'active-document'; target: ShaderNavigationTarget }
  | { kind: 'external-document'; target: ShaderNavigationTarget };

function normalizePath(path: string): string {
  return path.trim().replaceAll('\\', '/');
}

function comparablePath(path: string): string {
  return normalizePath(path).replace(/^\.\//, '');
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

/**
 * Classify a resolved diagnostic before the editor reveals it. Diagnostics for
 * includes or other source files must not be revealed in the currently active
 * editor, because doing so would highlight the right line in the wrong file.
 */
export function routeShaderDiagnostic(
  diagnostic: ShaderDiagnosticLocation,
  activePath?: string,
): ShaderDiagnosticRoute | undefined {
  const target = resolveShaderDiagnosticTarget(diagnostic, activePath);
  if (!target) return undefined;

  const currentPath = activePath ? comparablePath(activePath) : '';
  const targetPath = comparablePath(target.path);
  if (currentPath && targetPath === currentPath) {
    return { kind: 'active-document', target };
  }

  return { kind: 'external-document', target };
}

export function shaderDiagnosticTargetKey(target: ShaderNavigationTarget): string {
  return `${target.path}:${target.line}:${target.column}`;
}
