import type { EditorSettingsSnapshot } from '../../../common/editorWorkflowTypes';

const requiredSettings: Record<string, readonly string[]> = {
  windows: [
    'platform.windows.msvcToolchainPath',
    'platform.windows.sdkPath',
    'platform.windows.cmakePath',
    'platform.windows.ninjaPath',
  ],
  linux: ['platform.linux.compilerPath', 'platform.linux.sysrootPath'],
  macos: ['platform.apple.xcodePath', 'platform.apple.macosSdkPath'],
  ios: ['platform.apple.xcodePath', 'platform.apple.iosDeviceSdkPath'],
  android: ['platform.android.javaHome', 'platform.android.sdkPath', 'platform.android.ndkPath'],
  xbox: ['platform.xbox.gdkPath'],
  playstation: ['platform.playstation.sdkPath'],
  switch: ['platform.switch.sdkPath'],
};

export function platformUnavailableReason(platform: string, settings: EditorSettingsSnapshot | null): string {
  if (!settings) return 'Local setup not verified';
  const required = requiredSettings[platform];
  if (!required) return 'Local toolchain is unsupported';
  const missing = required.filter((key) => !settings.pathValidation?.[key]?.valid);
  return missing.map((key) => settings.pathValidation?.[key]?.message || `Configure ${key}`).join('; ');
}
