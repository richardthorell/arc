export type AiProviderId = 'openai' | 'anthropic';

export type AiProviderConnectionStatus = 'disconnected' | 'cold' | 'connected' | 'invalid';

export type AiProviderAccountStatus = {
  id: AiProviderId;
  label: string;
  connected: boolean;
  connectionStatus: AiProviderConnectionStatus;
};

export type AiProviderAccountsSnapshot = {
  secureStorageAvailable: boolean;
  secureStorageDetail?: string;
  providers: AiProviderAccountStatus[];
};

export type AiProviderTestAction = {
  action: 'test';
};
