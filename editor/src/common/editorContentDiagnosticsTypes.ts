/** Safe main-process diagnostics for a project-scoped cooked-content listener.
 * No bearer tokens, source paths, headers, URLs with secrets, or raw errors.
 */
export type EditorContentFailureCode =
  | 'bind-failed'
  | 'server-failed'
  | 'publish-rejected'
  | 'unauthorized'
  | 'forbidden'
  | 'rate-limited'
  | 'invalid-request'
  | 'not-found'
  | 'range-rejected'
  | 'content-changed'
  | 'unavailable';

export type EditorContentServerSnapshot = Readonly<{
  state: 'stopped' | 'listening';
  origin: string | null;
  port: number | null;
  bindMode: 'loopback';
  projectGuid: string;
  sessionId: string;
  profiles: ReadonlyArray<{ profile: string; family: string; generation: number }>;
  changeFeeds: ReadonlyArray<{ profile: string; oldestSequence: number; newestSequence: number }>;
  activeClients: number;
  activeTransfers: number;
  maximumConnections: number;
  requestsPerSecondLimit: number;
  requests: number;
  bytesSent: number;
  rejectedRequests: number;
  objectCacheHits: number;
  objectResponses: number;
  releaseResponses: number;
  feedResponses: number;
  snapshotResets: number;
  publicationFailures: number;
  lastFailure: Readonly<{ code: EditorContentFailureCode; status: number | null; atMs: number }> | null;
}>;
