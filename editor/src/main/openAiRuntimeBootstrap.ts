import type { AiInstructionProjectScope } from '../common/aiInstructionTypes';

let credentialSource: (() => string | null) | null = null;
let projectSource: (() => AiInstructionProjectScope | null) | null = null;
let installStarted = false;

export const registerOpenAiRuntimeCredentialSource = (
  source: () => string | null,
  activeProject?: () => AiInstructionProjectScope | null,
): void => {
  credentialSource = source;
  projectSource = activeProject ?? projectSource;
  if (installStarted || !process.versions.electron) return;
  installStarted = true;
  void import('./openAiRuntimeIpc')
    .then(({ installOpenAiRuntimeIpc }) => {
      installOpenAiRuntimeIpc(
        () => credentialSource?.() ?? null,
        () => projectSource?.() ?? null,
      );
    })
    .catch(() => {
      installStarted = false;
    });
};
