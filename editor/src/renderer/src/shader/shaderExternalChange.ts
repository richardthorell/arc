export type ShaderExternalChangeDecision =
  | { kind: 'unchanged' }
  | { kind: 'reload'; source: string; modifiedAt: string }
  | { kind: 'conflict'; source: string; modifiedAt: string };

export type ShaderExternalChangeInput = {
  localSource: string;
  confirmedSource: string;
  confirmedModifiedAt: string;
  diskSource: string;
  diskModifiedAt: string;
};

/**
 * Classifies a newly observed shader file without mutating editor state.
 *
 * Disk changes may be adopted automatically only while the document still
 * matches its last confirmed contents. Once the user has local edits, an
 * external change becomes an explicit conflict so callers can ask the user
 * which version to keep instead of silently overwriting either side.
 */
export const classifyShaderExternalChange = (input: ShaderExternalChangeInput): ShaderExternalChangeDecision => {
  const diskChanged =
    input.diskSource !== input.confirmedSource ||
    (input.confirmedModifiedAt.length > 0 && input.diskModifiedAt !== input.confirmedModifiedAt);

  if (!diskChanged) return { kind: 'unchanged' };

  const hasLocalChanges = input.localSource !== input.confirmedSource;
  if (hasLocalChanges) {
    return {
      kind: 'conflict',
      source: input.diskSource,
      modifiedAt: input.diskModifiedAt,
    };
  }

  return {
    kind: 'reload',
    source: input.diskSource,
    modifiedAt: input.diskModifiedAt,
  };
};
