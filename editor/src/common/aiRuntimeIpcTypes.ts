import type { AiProviderId } from './aiProviderTypes';
import type { AiRuntimeStreamEvent, AiSerializedRuntimeRequest } from './aiRuntimeTypes';

export type AiRuntimeStreamStartRequest = {
  requestId: string;
  providerId: AiProviderId;
  modelId: string;
  request: AiSerializedRuntimeRequest;
};

export type AiRuntimeStreamEnvelope = {
  requestId: string;
  event: AiRuntimeStreamEvent;
};
