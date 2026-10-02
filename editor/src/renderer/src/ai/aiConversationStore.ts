import {
  AI_CONVERSATION_STORE_VERSION,
  type AiConversationContextReference,
  type AiConversationMessage,
  type AiConversationStoreSnapshot,
  type AiConversationToolReference,
  type AiConversationUiState,
  type AiStoredConversation,
} from '../../../common/aiConversationTypes';

export const legacyAiConversationStorageKey = 'arc.ai.conversations.v1';
export const legacyAiConversationMigrationKey = 'arc.ai.conversations.project-migration.v1';

const normalizeProjectGuid = (projectGuid: string) => projectGuid.trim().toLocaleLowerCase();

export const aiProjectConversationStorageKey = (projectGuid: string) =>
  `arc.ai.projects.${normalizeProjectGuid(projectGuid)}.conversations.v${AI_CONVERSATION_STORE_VERSION}`;

const isOptionalString = (value: unknown): value is string | undefined =>
  value === undefined || typeof value === 'string';

const isContextReference = (value: unknown): value is AiConversationContextReference => {
  if (!value || typeof value !== 'object') return false;
  const reference = value as Partial<AiConversationContextReference>;
  return (
    typeof reference.id === 'string' &&
    typeof reference.kind === 'string' &&
    isOptionalString(reference.label) &&
    isOptionalString(reference.stableId) &&
    (reference.metadata === undefined || (reference.metadata !== null && typeof reference.metadata === 'object'))
  );
};

const isToolReference = (value: unknown): value is AiConversationToolReference => {
  if (!value || typeof value !== 'object') return false;
  const reference = value as Partial<AiConversationToolReference>;
  return (
    typeof reference.toolCallId === 'string' &&
    typeof reference.name === 'string' &&
    (reference.state === 'pending' ||
      reference.state === 'complete' ||
      reference.state === 'error' ||
      reference.state === 'cancelled') &&
    isOptionalString(reference.summary)
  );
};

const isMessage = (value: unknown): value is AiConversationMessage => {
  if (!value || typeof value !== 'object') return false;
  const message = value as Partial<AiConversationMessage>;
  return (
    typeof message.id === 'string' &&
    (message.role === 'user' || message.role === 'assistant' || message.role === 'system') &&
    typeof message.content === 'string' &&
    typeof message.createdAt === 'string' &&
    (message.state === 'complete' || message.state === 'streaming' || message.state === 'error') &&
    isOptionalString(message.modelId) &&
    isOptionalString(message.modelLabel) &&
    (message.contextReferences === undefined ||
      (Array.isArray(message.contextReferences) && message.contextReferences.every(isContextReference))) &&
    (message.toolReferences === undefined ||
      (Array.isArray(message.toolReferences) && message.toolReferences.every(isToolReference)))
  );
};

const isConversation = (value: unknown): value is AiStoredConversation => {
  if (!value || typeof value !== 'object') return false;
  const conversation = value as Partial<AiStoredConversation>;
  return (
    typeof conversation.id === 'string' &&
    typeof conversation.title === 'string' &&
    typeof conversation.createdAt === 'string' &&
    typeof conversation.updatedAt === 'string' &&
    isOptionalString(conversation.modelId) &&
    isOptionalString(conversation.modelLabel) &&
    Array.isArray(conversation.messages) &&
    conversation.messages.every(isMessage) &&
    (conversation.summary === undefined ||
      (typeof conversation.summary.text === 'string' &&
        typeof conversation.summary.updatedAt === 'string' &&
        isOptionalString(conversation.summary.throughMessageId))) &&
    (conversation.pinnedContext === undefined ||
      (Array.isArray(conversation.pinnedContext) && conversation.pinnedContext.every(isContextReference)))
  );
};

const normalizeConversation = (conversation: AiStoredConversation): AiStoredConversation => ({
  ...conversation,
  messages: conversation.messages.map((message) =>
    message.state === 'streaming' ? { ...message, state: 'error' as const } : { ...message },
  ),
});

const normalizeUiState = (value: unknown): AiConversationUiState => {
  if (!value || typeof value !== 'object') return {};
  const source = value as Partial<AiConversationUiState>;
  return {
    ...(typeof source.activeConversationId === 'string' ? { activeConversationId: source.activeConversationId } : {}),
    ...(typeof source.selectedModelId === 'string' ? { selectedModelId: source.selectedModelId } : {}),
  };
};

export const createEmptyAiConversationStore = (projectGuid: string): AiConversationStoreSnapshot => ({
  schemaVersion: AI_CONVERSATION_STORE_VERSION,
  projectGuid: normalizeProjectGuid(projectGuid),
  updatedAt: new Date(0).toISOString(),
  conversations: [],
  uiState: {},
});

export const migrateAiConversationStoreDocument = (
  value: unknown,
  projectGuid: string,
): AiConversationStoreSnapshot | null => {
  const normalizedGuid = normalizeProjectGuid(projectGuid);

  // Legacy renderer storage was a bare conversation array with no project identity.
  if (Array.isArray(value)) {
    const conversations = value.filter(isConversation).filter((conversation) => conversation.messages.length > 0);
    return {
      schemaVersion: AI_CONVERSATION_STORE_VERSION,
      projectGuid: normalizedGuid,
      updatedAt: new Date().toISOString(),
      conversations: conversations.map(normalizeConversation),
      uiState: {},
    };
  }

  if (!value || typeof value !== 'object') return null;
  const source = value as Partial<AiConversationStoreSnapshot> & { schemaVersion?: number };
  if (source.schemaVersion !== AI_CONVERSATION_STORE_VERSION) return null;
  if (typeof source.projectGuid !== 'string' || normalizeProjectGuid(source.projectGuid) !== normalizedGuid)
    return null;
  if (!Array.isArray(source.conversations) || !source.conversations.every(isConversation)) return null;

  const conversations = source.conversations
    .filter((conversation) => conversation.messages.length > 0)
    .map(normalizeConversation);
  const uiState = normalizeUiState(source.uiState);
  if (
    uiState.activeConversationId &&
    !conversations.some((conversation) => conversation.id === uiState.activeConversationId)
  ) {
    delete uiState.activeConversationId;
  }

  return {
    schemaVersion: AI_CONVERSATION_STORE_VERSION,
    projectGuid: normalizedGuid,
    updatedAt: typeof source.updatedAt === 'string' ? source.updatedAt : new Date().toISOString(),
    conversations,
    uiState,
  };
};

type ConversationStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

export const loadAiConversationStore = (
  projectGuid: string,
  storage: ConversationStorage = localStorage,
): AiConversationStoreSnapshot => {
  const normalizedGuid = normalizeProjectGuid(projectGuid);
  if (!normalizedGuid) return createEmptyAiConversationStore(projectGuid);

  const projectKey = aiProjectConversationStorageKey(normalizedGuid);
  try {
    const raw = storage.getItem(projectKey);
    if (raw) {
      const migrated = migrateAiConversationStoreDocument(JSON.parse(raw) as unknown, normalizedGuid);
      if (migrated) return migrated;
    }

    // The old renderer store was global. Import it exactly once into the project
    // that is active during migration, then remove the global source so it can
    // never leak into another project.
    if (!storage.getItem(legacyAiConversationMigrationKey)) {
      const legacyRaw = storage.getItem(legacyAiConversationStorageKey);
      if (legacyRaw) {
        const migrated = migrateAiConversationStoreDocument(JSON.parse(legacyRaw) as unknown, normalizedGuid);
        if (migrated) {
          storage.setItem(projectKey, JSON.stringify(migrated));
          storage.removeItem(legacyAiConversationStorageKey);
          storage.setItem(legacyAiConversationMigrationKey, normalizedGuid);
          return migrated;
        }
      }
      storage.setItem(legacyAiConversationMigrationKey, normalizedGuid);
    }
  } catch {
    // Corrupt user-private state should never prevent the editor from opening.
  }

  return createEmptyAiConversationStore(normalizedGuid);
};

export const saveAiConversationStore = (
  projectGuid: string,
  conversations: readonly AiStoredConversation[],
  uiState: AiConversationUiState,
  storage: ConversationStorage = localStorage,
): AiConversationStoreSnapshot => {
  const normalizedGuid = normalizeProjectGuid(projectGuid);
  const snapshot: AiConversationStoreSnapshot = {
    schemaVersion: AI_CONVERSATION_STORE_VERSION,
    projectGuid: normalizedGuid,
    updatedAt: new Date().toISOString(),
    conversations: conversations
      .filter((conversation) => conversation.messages.length > 0)
      .map((conversation) => normalizeConversation(conversation)),
    uiState: { ...uiState },
  };
  storage.setItem(aiProjectConversationStorageKey(normalizedGuid), JSON.stringify(snapshot));
  return snapshot;
};
