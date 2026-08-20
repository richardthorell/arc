import type { EditorDocumentKind, EditorRegistry } from './editorTypes';

export const createEditorRegistry = (registrations: EditorRegistry): EditorRegistry => registrations;

export const getEditorRegistration = (registry: EditorRegistry, kind: EditorDocumentKind) => registry[kind];
