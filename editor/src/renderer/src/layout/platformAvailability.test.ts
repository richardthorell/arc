import { describe, expect, it } from 'vitest';
import type { EditorSettingsSnapshot } from '../../../common/editorWorkflowTypes';
import { platformUnavailableReason } from './platformAvailability';

describe('local platform availability', () => {
  it.each(['windows', 'linux', 'macos', 'ios', 'android', 'xbox', 'playstation', 'switch'])(
    'requires validated local settings for %s',
    (platform) => {
      expect(platformUnavailableReason(platform, null)).toBeTruthy();
      expect(
        platformUnavailableReason(platform, {
          values: { 'platform.android.sdkPath': '/entered-only' },
        } as unknown as EditorSettingsSnapshot),
      ).toBeTruthy();
    },
  );
  it('accepts detected SDKs but rejects incomplete Android setups with their diagnostic', () => {
    const settings = {
      pathValidation: {
        'platform.android.javaHome': { valid: true },
        'platform.android.sdkPath': { valid: true },
        'platform.android.ndkPath': { valid: false, message: 'NDK not detected' },
      },
    } as unknown as EditorSettingsSnapshot;
    expect(platformUnavailableReason('android', settings)).toBe('NDK not detected');
    settings.pathValidation!['platform.android.ndkPath'].valid = true;
    expect(platformUnavailableReason('android', settings)).toBe('');
  });
});
