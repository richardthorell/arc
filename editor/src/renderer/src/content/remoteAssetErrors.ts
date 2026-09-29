export type RemoteAssetOperation = 'search' | 'manifest' | 'import';

const operationLabel: Record<RemoteAssetOperation, string> = {
  search: 'browse remote assets',
  manifest: 'load download variants',
  import: 'import this asset',
};

const reasonMessage = (reason: unknown): string => {
  if (reason instanceof Error) return reason.message.trim();
  if (typeof reason === 'string') return reason.trim();
  return '';
};

/**
 * Convert provider/IPC failures into actionable user-facing messages without
 * leaking transport details into the Remote Asset Browser UI.
 */
export function remoteAssetFailureMessage(operation: RemoteAssetOperation, reason: unknown): string {
  const message = reasonMessage(reason);
  const prefix = `Unable to ${operationLabel[operation]}.`;
  if (!message) return `${prefix} Try again, or check the asset provider connection.`;

  const normalized = message.toLowerCase();
  if (normalized.includes('checksum') || normalized.includes('hash mismatch')) {
    return `${prefix} The downloaded file failed integrity validation. Try the download again.`;
  }
  if (normalized.includes('timeout') || normalized.includes('timed out')) {
    return `${prefix} The asset provider timed out. Check your connection and try again.`;
  }
  if (normalized.includes('network') || normalized.includes('fetch failed') || normalized.includes('econn')) {
    return `${prefix} The asset provider could not be reached. Check your connection and try again.`;
  }

  return `${prefix} ${message}`;
}
