import type { AiRuntimeRequest, AiRuntimeStreamEvent } from '../common/aiRuntimeTypes';

export type OpenAiRuntimeSettings = {
  credential: string;
  modelId: string;
  reasoningEffort?: string;
  organizationId?: string;
  projectId?: string;
  storeResponses?: boolean;
};

export class OpenAiRuntimeAdapter {
  async *stream(_request: AiRuntimeRequest, _settings: OpenAiRuntimeSettings): AsyncGenerator<AiRuntimeStreamEvent> {
    yield { type: 'error', code: 'provider', message: 'OpenAI runtime adapter is not initialized' };
  }
}
