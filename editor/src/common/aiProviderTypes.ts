export type AiProviderId = 'openai' | 'anthropic';

export type AiProviderAccountStatus = {
  id: AiProviderId;
  label: string;
  connected: boolean;
};

export type AiProviderAccountsSnapshot = {
  secureStorageAvailable: boolean;
  secureStorageDetail?: string;
  providers: AiProviderAccountStatus[];
};
