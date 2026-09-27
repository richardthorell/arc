import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import type { EditorPathValidation } from '../common/editorWorkflowTypes';

export const androidToolchainSettingKeys = {
  javaHome: 'platform.android.javaHome',
  sdkPath: 'platform.android.sdkPath',
  ndkPath: 'platform.android.ndkPath',
} as const;

type AndroidToolchainPreferences = {
  javaHome?: string;
  sdkPath?: string;
  ndkPath?: string;
};

type Environment = Record<string, string | undefined>;

type Candidate = {
  path: string;
  source: EditorPathValidation['source'];
};

const fileExists = (filePath: string): boolean => {
  try {
    return fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
};

const directoryExists = (directoryPath: string): boolean => {
  try {
    return fs.statSync(directoryPath).isDirectory();
  } catch {
    return false;
  }
};

const uniqueCandidates = (candidates: Array<Candidate | null>): Candidate[] => {
  const seen = new Set<string>();
  return candidates.filter((candidate): candidate is Candidate => {
    if (!candidate?.path) return false;
    const normalized = path.resolve(candidate.path);
    if (seen.has(normalized)) return false;
    seen.add(normalized);
    candidate.path = normalized;
    return true;
  });
};

const candidate = (
  value: string | undefined,
  source: EditorPathValidation['source'],
): Candidate | null => (value?.trim() ? { path: value.trim(), source } : null);

const executableName = (name: string): string => (process.platform === 'win32' ? `${name}.exe` : name);

const javaHomeFromPath = (environment: Environment): string | null => {
  const pathValue = environment.PATH ?? environment.Path ?? environment.path;
  if (!pathValue) return null;
  for (const entry of pathValue.split(path.delimiter)) {
    if (!entry) continue;
    const javaPath = path.join(entry, executableName('java'));
    if (fileExists(javaPath)) return path.dirname(entry);
  }
  return null;
};

const defaultJavaHomes = (environment: Environment): string[] => {
  if (process.platform === 'win32') {
    return [
      environment.ProgramFiles ? path.join(environment.ProgramFiles, 'Android', 'Android Studio', 'jbr') : '',
      environment['ProgramFiles(x86)']
        ? path.join(environment['ProgramFiles(x86)'], 'Android', 'Android Studio', 'jbr')
        : '',
    ].filter(Boolean);
  }
  if (process.platform === 'darwin') return ['/Applications/Android Studio.app/Contents/jbr/Contents/Home'];
  return ['/opt/android-studio/jbr', path.join(os.homedir(), 'android-studio', 'jbr')];
};

const defaultSdkRoots = (environment: Environment): string[] => {
  if (process.platform === 'win32')
    return [environment.LOCALAPPDATA ? path.join(environment.LOCALAPPDATA, 'Android', 'Sdk') : ''].filter(Boolean);
  if (process.platform === 'darwin') return [path.join(os.homedir(), 'Library', 'Android', 'sdk')];
  return [path.join(os.homedir(), 'Android', 'Sdk'), path.join(os.homedir(), 'Android', 'sdk')];
};

const readProperty = (filePath: string, property: string): string | null => {
  try {
    const text = fs.readFileSync(filePath, 'utf8');
    const match = text.match(new RegExp(`^${property.replaceAll('.', '\\.')}\\s*=\\s*(.+)$`, 'm'));
    return match?.[1]?.trim() ?? null;
  } catch {
    return null;
  }
};

const readJavaVersion = (javaHome: string): string | null => {
  try {
    const text = fs.readFileSync(path.join(javaHome, 'release'), 'utf8');
    return text.match(/^JAVA_VERSION="?([^"\r\n]+)"?/m)?.[1] ?? null;
  } catch {
    return null;
  }
};

const validateJavaHome = (candidatePath: Candidate | null): EditorPathValidation => {
  if (!candidatePath)
    return { valid: false, resolvedPath: '', message: 'Java/JDK not detected', source: 'unresolved' };
  const java = path.join(candidatePath.path, 'bin', executableName('java'));
  const javac = path.join(candidatePath.path, 'bin', executableName('javac'));
  if (!fileExists(java) || !fileExists(javac)) {
    return {
      valid: false,
      resolvedPath: candidatePath.path,
      message: 'Path is not a valid JDK',
      source: candidatePath.source,
    };
  }
  const version = readJavaVersion(candidatePath.path);
  return {
    valid: true,
    resolvedPath: candidatePath.path,
    message: version ? `Validated · Java ${version}` : 'Validated · Java/JDK',
    source: candidatePath.source,
  };
};

const validateSdkRoot = (candidatePath: Candidate | null): EditorPathValidation => {
  if (!candidatePath)
    return { valid: false, resolvedPath: '', message: 'Android SDK not detected', source: 'unresolved' };
  const adb = path.join(candidatePath.path, 'platform-tools', executableName('adb'));
  if (!fileExists(adb)) {
    return {
      valid: false,
      resolvedPath: candidatePath.path,
      message: 'Android SDK is missing platform-tools/adb',
      source: candidatePath.source,
    };
  }
  return {
    valid: true,
    resolvedPath: candidatePath.path,
    message: 'Validated · Android SDK / ADB',
    source: candidatePath.source,
  };
};

const versionParts = (value: string): number[] => value.split(/[.-]/).map((part) => Number.parseInt(part, 10) || 0);

const compareVersionsDescending = (left: string, right: string): number => {
  const a = versionParts(left);
  const b = versionParts(right);
  const count = Math.max(a.length, b.length);
  for (let index = 0; index < count; ++index) {
    const delta = (b[index] ?? 0) - (a[index] ?? 0);
    if (delta !== 0) return delta;
  }
  return right.localeCompare(left);
};

const newestNdkUnderSdk = (sdkRoot: string): string | null => {
  const ndkRoot = path.join(sdkRoot, 'ndk');
  if (directoryExists(ndkRoot)) {
    try {
      const versions = fs
        .readdirSync(ndkRoot, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name)
        .sort(compareVersionsDescending);
      if (versions[0]) return path.join(ndkRoot, versions[0]);
    } catch {
      // Fall through to the legacy location.
    }
  }
  const legacy = path.join(sdkRoot, 'ndk-bundle');
  return directoryExists(legacy) ? legacy : null;
};

const validateNdkRoot = (candidatePath: Candidate | null): EditorPathValidation => {
  if (!candidatePath)
    return { valid: false, resolvedPath: '', message: 'Android NDK not detected', source: 'unresolved' };
  const properties = path.join(candidatePath.path, 'source.properties');
  const toolchain = path.join(candidatePath.path, 'build', 'cmake', 'android.toolchain.cmake');
  if (!fileExists(properties) || !fileExists(toolchain)) {
    return {
      valid: false,
      resolvedPath: candidatePath.path,
      message: 'Path is not a valid Android NDK',
      source: candidatePath.source,
    };
  }
  const version = readProperty(properties, 'Pkg.Revision');
  return {
    valid: true,
    resolvedPath: candidatePath.path,
    message: version ? `Validated · NDK ${version}` : 'Validated · Android NDK',
    source: candidatePath.source,
  };
};

const firstValidOrFirst = (
  candidates: Candidate[],
  validate: (value: Candidate | null) => EditorPathValidation,
): Candidate | null => {
  for (const value of candidates) if (validate(value).valid) return value;
  return candidates[0] ?? null;
};

export const resolveAndroidToolchainValidation = (
  preferences: AndroidToolchainPreferences,
  environment: Environment = process.env,
): Record<string, EditorPathValidation> => {
  const javaCandidates = uniqueCandidates([
    candidate(preferences.javaHome, 'configured'),
    candidate(environment.JAVA_HOME, 'environment'),
    candidate(javaHomeFromPath(environment), 'environment'),
    ...defaultJavaHomes(environment).map((value) => candidate(value, 'default')),
  ]);
  const sdkCandidates = uniqueCandidates([
    candidate(preferences.sdkPath, 'configured'),
    candidate(environment.ANDROID_SDK_ROOT, 'environment'),
    candidate(environment.ANDROID_HOME, 'environment'),
    ...defaultSdkRoots(environment).map((value) => candidate(value, 'default')),
  ]);

  const javaCandidate = firstValidOrFirst(javaCandidates, validateJavaHome);
  const sdkCandidate = firstValidOrFirst(sdkCandidates, validateSdkRoot);
  const ndkFromSdk = sdkCandidate ? newestNdkUnderSdk(sdkCandidate.path) : null;
  const ndkCandidates = uniqueCandidates([
    candidate(preferences.ndkPath, 'configured'),
    candidate(environment.ANDROID_NDK_HOME, 'environment'),
    candidate(environment.ANDROID_NDK_ROOT, 'environment'),
    candidate(ndkFromSdk ?? undefined, 'derived'),
  ]);
  const ndkCandidate = firstValidOrFirst(ndkCandidates, validateNdkRoot);

  return {
    [androidToolchainSettingKeys.javaHome]: validateJavaHome(javaCandidate),
    [androidToolchainSettingKeys.sdkPath]: validateSdkRoot(sdkCandidate),
    [androidToolchainSettingKeys.ndkPath]: validateNdkRoot(ndkCandidate),
  };
};
