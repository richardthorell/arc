import type { AiInstructionProjectScope } from '../common/aiInstructionTypes';
import { registerOpenAiRuntimeCredentialSource } from './openAiRuntimeBootstrap';
import {
  AiProviderService as CoreAiProviderService,
  type AiProviderCredentialValidator,
  type AiProviderSecureStorage,
} from './aiProviderServiceCore';

export type { AiProviderCredentialValidator, AiProviderSecureStorage } from './aiProviderServiceCore';

export class AiProviderService extends CoreAiProviderService {
  constructor(
    storagePath: string,
    secureStorage?: () => AiProviderSecureStorage,
    validateCredential?: AiProviderCredentialValidator,
    activeProject?: () => AiInstructionProjectScope | null,
  ) {
    super(storagePath, secureStorage, validateCredential);
    registerOpenAiRuntimeCredentialSource(() => this.credential('openai'), activeProject);
  }
}
