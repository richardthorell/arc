export type EditorReferenceKind = 'entity' | 'asset' | 'scene';

export type EditorReference = {
  kind: EditorReferenceKind;
  id: string;
};

export type ResolvedEditorReference = {
  reference: EditorReference;
  label: string;
  subtitle?: string;
  thumbnailUrl?: string | null;
  disabled?: boolean;
};

export type EditorReferenceController = {
  resolve: (reference: EditorReference) => ResolvedEditorReference | null | Promise<ResolvedEditorReference | null>;
  activate: (reference: EditorReference) => void | Promise<void>;
  focus?: (reference: EditorReference) => void | Promise<void>;
  highlight?: (reference: EditorReference, active: boolean) => void | Promise<void>;
};

const editorReferencePattern = /^arc:\/\/(entity|asset|scene)\/([^/?#]+)$/u;

export const parseEditorReference = (value: string): EditorReference | null => {
  const match = editorReferencePattern.exec(value.trim());
  if (!match) return null;

  try {
    const id = decodeURIComponent(match[2]);
    if (!id.trim()) return null;
    return { kind: match[1] as EditorReferenceKind, id };
  } catch {
    return null;
  }
};

export const editorReferenceUri = (reference: EditorReference): string =>
  `arc://${reference.kind}/${encodeURIComponent(reference.id)}`;

export type EditorReferenceHandlers = {
  resolveEntity?: (
    id: string,
  ) => Omit<ResolvedEditorReference, 'reference'> | null | Promise<Omit<ResolvedEditorReference, 'reference'> | null>;
  resolveAsset?: (
    id: string,
  ) => Omit<ResolvedEditorReference, 'reference'> | null | Promise<Omit<ResolvedEditorReference, 'reference'> | null>;
  resolveScene?: (
    id: string,
  ) => Omit<ResolvedEditorReference, 'reference'> | null | Promise<Omit<ResolvedEditorReference, 'reference'> | null>;
  activateEntity?: (id: string) => void | Promise<void>;
  activateAsset?: (id: string) => void | Promise<void>;
  activateScene?: (id: string) => void | Promise<void>;
  focusEntity?: (id: string) => void | Promise<void>;
  focusAsset?: (id: string) => void | Promise<void>;
  focusScene?: (id: string) => void | Promise<void>;
  highlightEntity?: (id: string, active: boolean) => void | Promise<void>;
  highlightAsset?: (id: string, active: boolean) => void | Promise<void>;
  highlightScene?: (id: string, active: boolean) => void | Promise<void>;
};

const handlerFor = <T>(reference: EditorReference, handlers: Partial<Record<EditorReferenceKind, T>>): T | undefined =>
  handlers[reference.kind];

export const createEditorReferenceController = (handlers: EditorReferenceHandlers): EditorReferenceController => ({
  resolve: async (reference) => {
    const resolve = handlerFor(reference, {
      entity: handlers.resolveEntity,
      asset: handlers.resolveAsset,
      scene: handlers.resolveScene,
    });
    if (!resolve) return null;
    const resolved = await resolve(reference.id);
    return resolved ? { ...resolved, reference } : null;
  },
  activate: async (reference) => {
    const activate = handlerFor(reference, {
      entity: handlers.activateEntity,
      asset: handlers.activateAsset,
      scene: handlers.activateScene,
    });
    await activate?.(reference.id);
  },
  focus: async (reference) => {
    const focus = handlerFor(reference, {
      entity: handlers.focusEntity,
      asset: handlers.focusAsset,
      scene: handlers.focusScene,
    });
    if (focus) await focus(reference.id);
    else
      await handlerFor(reference, {
        entity: handlers.activateEntity,
        asset: handlers.activateAsset,
        scene: handlers.activateScene,
      })?.(reference.id);
  },
  highlight: async (reference, active) => {
    const highlight = handlerFor(reference, {
      entity: handlers.highlightEntity,
      asset: handlers.highlightAsset,
      scene: handlers.highlightScene,
    });
    await highlight?.(reference.id, active);
  },
});
