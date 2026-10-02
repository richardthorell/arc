let credentialSource: (() => string | null) | null = null;
let installStarted = false;

export const registerOpenAiRuntimeCredentialSource = (source: () => string | null): void => {
  credentialSource = source;
  if (installStarted || !process.versions.electron) return;
  installStarted = true;
  void import('./openAiRuntimeIpc')
    .then(({ installOpenAiRuntimeIpc }) => {
      installOpenAiRuntimeIpc(() => credentialSource?.() ?? null);
    })
    .catch(() => {
      installStarted = false;
    });
};
