export type StandaloneBuildConfiguration = 'debug' | 'development' | 'release';

export type StandaloneLaunchRequest = Readonly<{
  version: 1;
  sessionId: string;
  projectPath: string;
  configuration: StandaloneBuildConfiguration;
}>;

export type StandaloneLaunchFailureCode = 'invalid-project' | 'runtime-missing' | 'build-required' | 'launch-failed';

export type StandaloneLaunchFailure = Readonly<{
  code: StandaloneLaunchFailureCode;
  message: string;
}>;

export type StandaloneSessionStatus = 'launching' | 'running' | 'stopping' | 'stopped' | 'failed';

export type StandaloneSession = Readonly<{
  request: StandaloneLaunchRequest;
  status: StandaloneSessionStatus;
  processId?: number;
  failure?: StandaloneLaunchFailure;
}>;

export type StandaloneSessionEvent =
  | Readonly<{ type: 'started'; sessionId: string; processId: number }>
  | Readonly<{ type: 'stop-requested'; sessionId: string }>
  | Readonly<{ type: 'stopped'; sessionId: string }>
  | Readonly<{ type: 'failed'; sessionId: string; failure: StandaloneLaunchFailure }>;

const normalizeProjectPath = (projectPath: string): string => projectPath.trim().replace(/\\/g, '/');

/**
 * Creates the editor-to-host contract for a standalone runtime launch.
 *
 * The session identity is supplied by the editor and survives process launch so
 * logs, failures, stop, and relaunch can all target the same standalone session.
 * The project path identifies packaged/runtime input; the contract deliberately
 * carries no editor-world or editor-host state.
 */
export const createStandaloneLaunchRequest = (
  sessionId: string,
  projectPath: string,
  configuration: StandaloneBuildConfiguration,
): StandaloneLaunchRequest => {
  const normalizedSessionId = sessionId.trim();
  const normalizedProjectPath = normalizeProjectPath(projectPath);

  if (!normalizedSessionId) throw new Error('Standalone launch requires a session ID');
  if (!normalizedProjectPath) throw new Error('Standalone launch requires a project path');

  return {
    version: 1,
    sessionId: normalizedSessionId,
    projectPath: normalizedProjectPath,
    configuration,
  };
};

export const createStandaloneSession = (request: StandaloneLaunchRequest): StandaloneSession => ({
  request,
  status: 'launching',
});

/**
 * Applies host lifecycle events only to the standalone session they identify.
 * Stale events from a stopped/relaunched process cannot mutate a newer session.
 */
export const reduceStandaloneSession = (
  session: StandaloneSession,
  event: StandaloneSessionEvent,
): StandaloneSession => {
  if (event.sessionId !== session.request.sessionId) return session;

  switch (event.type) {
    case 'started':
      if (session.status !== 'launching' || !Number.isInteger(event.processId) || event.processId <= 0) return session;
      return { ...session, status: 'running', processId: event.processId, failure: undefined };
    case 'stop-requested':
      if (session.status !== 'launching' && session.status !== 'running') return session;
      return { ...session, status: 'stopping' };
    case 'stopped':
      if (session.status === 'stopped') return session;
      return { ...session, status: 'stopped', processId: undefined };
    case 'failed':
      if (session.status === 'stopped') return session;
      return { ...session, status: 'failed', processId: undefined, failure: event.failure };
  }
};

export const createStandaloneRelaunchRequest = (
  session: StandaloneSession,
  nextSessionId: string,
): StandaloneLaunchRequest =>
  createStandaloneLaunchRequest(nextSessionId, session.request.projectPath, session.request.configuration);

export const describeStandaloneLaunchFailure = (failure: StandaloneLaunchFailure): string => {
  const detail = failure.message.trim();

  switch (failure.code) {
    case 'invalid-project':
      return detail || 'The project cannot be launched standalone.';
    case 'runtime-missing':
      return detail || 'The standalone runtime is not available for this project.';
    case 'build-required':
      return detail || 'Build the project before launching it standalone.';
    case 'launch-failed':
      return detail || 'The standalone runtime failed to start.';
  }
};
