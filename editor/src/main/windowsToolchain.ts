import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

import type { EditorPathValidation } from '../common/editorWorkflowTypes';

export const windowsToolchainSettingKeys = {
  visualStudioPath: 'platform.windows.visualStudioPath',
  msvcToolchainPath: 'platform.windows.msvcToolchainPath',
  sdkPath: 'platform.windows.sdkPath',
  cmakePath: 'platform.windows.cmakePath',
  ninjaPath: 'platform.windows.ninjaPath',
} as const;

type WindowsToolchainPreferences = {
  visualStudioPath?: string;
  msvcToolchainPath?: string;
  sdkPath?: string;
  cmakePath?: string;
  ninjaPath?: string;
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

const candidate = (value: string | null | undefined, source: EditorPathValidation['source']): Candidate | null =>
  value?.trim() ? { path: value.trim(), source } : null;

const uniqueCandidates = (candidates: Array<Candidate | null>): Candidate[] => {
  const seen = new Set<string>();
  return candidates.filter((value): value is Candidate => {
    if (!value?.path) return false;
    const normalized = path.resolve(value.path);
    const key = process.platform === 'win32' ? normalized.toLocaleLowerCase() : normalized;
    if (seen.has(key)) return false;
    seen.add(key);
    value.path = normalized;
    return true;
  });
};

const executableNames = (name: string): string[] =>
  process.platform === 'win32' ? [`${name}.exe`, name] : [name, `${name}.exe`];

const findExecutableOnPath = (name: string, environment: Environment): string | null => {
  const pathValue = environment.PATH ?? environment.Path ?? environment.path;
  if (!pathValue) return null;
  for (const entry of pathValue.split(path.delimiter)) {
    if (!entry) continue;
    for (const executable of executableNames(name)) {
      const executablePath = path.join(entry, executable);
      if (fileExists(executablePath)) return executablePath;
    }
  }
  return null;
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

const newestVersionDirectory = (root: string): string | null => {
  if (!directoryExists(root)) return null;
  try {
    const versions = fs
      .readdirSync(root, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && /^\d+(?:\.\d+)+/.test(entry.name))
      .map((entry) => entry.name)
      .sort(compareVersionsDescending);
    return versions[0] ? path.join(root, versions[0]) : null;
  } catch {
    return null;
  }
};

const validateVisualStudioRoot = (value: Candidate | null): EditorPathValidation => {
  if (!value)
    return { valid: false, resolvedPath: '', message: 'Visual Studio not detected', source: 'unresolved' };
  const vcvars = path.join(value.path, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat');
  if (!directoryExists(value.path) || !fileExists(vcvars)) {
    return {
      valid: false,
      resolvedPath: value.path,
      message: 'Path is not a Visual Studio C++ installation',
      source: value.source,
    };
  }
  return {
    valid: true,
    resolvedPath: value.path,
    message: 'Validated · Visual Studio C++',
    source: value.source,
  };
};

const validateMsvcToolchainRoot = (value: Candidate | null): EditorPathValidation => {
  if (!value)
    return { valid: false, resolvedPath: '', message: 'MSVC toolchain not detected', source: 'unresolved' };
  const compiler = path.join(value.path, 'bin', 'Hostx64', 'x64', 'cl.exe');
  if (!directoryExists(value.path) || !fileExists(compiler)) {
    return {
      valid: false,
      resolvedPath: value.path,
      message: 'Path is not a valid MSVC toolchain',
      source: value.source,
    };
  }
  return {
    valid: true,
    resolvedPath: value.path,
    message: `Validated · MSVC ${path.basename(value.path)}`,
    source: value.source,
  };
};

const validateWindowsSdkRoot = (value: Candidate | null): EditorPathValidation => {
  if (!value)
    return { valid: false, resolvedPath: '', message: 'Windows SDK not detected', source: 'unresolved' };
  const versionRoot = newestVersionDirectory(path.join(value.path, 'Include'));
  const hasHeaders = Boolean(versionRoot && fileExists(path.join(versionRoot, 'um', 'Windows.h')));
  if (!directoryExists(value.path) || !hasHeaders) {
    return {
      valid: false,
      resolvedPath: value.path,
      message: 'Path is not a valid Windows SDK',
      source: value.source,
    };
  }
  return {
    valid: true,
    resolvedPath: value.path,
    message: versionRoot ? `Validated · Windows SDK ${path.basename(versionRoot)}` : 'Validated · Windows SDK',
    source: value.source,
  };
};

const validateExecutable = (label: string, value: Candidate | null): EditorPathValidation => {
  if (!value) return { valid: false, resolvedPath: '', message: `${label} not detected`, source: 'unresolved' };
  if (!fileExists(value.path)) {
    return {
      valid: false,
      resolvedPath: value.path,
      message: `Path is not a valid ${label} executable`,
      source: value.source,
    };
  }
  return {
    valid: true,
    resolvedPath: value.path,
    message: `Validated · ${label}`,
    source: value.source,
  };
};

const visualStudioFromVcInstallDir = (environment: Environment): string | null => {
  const vcInstallDir = environment.VCINSTALLDIR?.trim();
  return vcInstallDir ? path.dirname(path.resolve(vcInstallDir)) : null;
};

const findVswhere = (environment: Environment): string | null => {
  const fromPath = findExecutableOnPath('vswhere', environment);
  if (fromPath) return fromPath;
  for (const root of [environment['ProgramFiles(x86)'], environment.ProgramFiles]) {
    if (!root) continue;
    const executable = path.join(root, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe');
    if (fileExists(executable)) return executable;
  }
  return null;
};

const visualStudioFromVswhere = (environment: Environment): string | null => {
  const vswhere = findVswhere(environment);
  if (!vswhere) return null;
  try {
    return (
      execFileSync(
        vswhere,
        [
          '-latest',
          '-products',
          '*',
          '-requires',
          'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
          '-property',
          'installationPath',
        ],
        { encoding: 'utf8', windowsHide: true },
      ).trim() || null
    );
  } catch {
    return null;
  }
};

const defaultVisualStudioRoots = (environment: Environment): string[] => {
  const roots: string[] = [];
  for (const programFiles of [environment.ProgramFiles, environment['ProgramFiles(x86)']]) {
    if (!programFiles) continue;
    const visualStudioRoot = path.join(programFiles, 'Microsoft Visual Studio');
    if (!directoryExists(visualStudioRoot)) continue;
    try {
      for (const version of fs.readdirSync(visualStudioRoot, { withFileTypes: true })) {
        if (!version.isDirectory() || version.name === 'Installer') continue;
        const versionRoot = path.join(visualStudioRoot, version.name);
        for (const edition of fs.readdirSync(versionRoot, { withFileTypes: true })) {
          if (edition.isDirectory()) roots.push(path.join(versionRoot, edition.name));
        }
      }
    } catch {
      // Keep successfully enumerated candidates.
    }
  }
  return roots;
};

const defaultWindowsSdkRoots = (environment: Environment): string[] =>
  [environment['ProgramFiles(x86)'], environment.ProgramFiles]
    .filter((value): value is string => Boolean(value))
    .map((value) => path.join(value, 'Windows Kits', '10'));

const msvcUnderVisualStudio = (visualStudioPath: string | null): string | null =>
  visualStudioPath ? newestVersionDirectory(path.join(visualStudioPath, 'VC', 'Tools', 'MSVC')) : null;

const bundledCmake = (visualStudioPath: string | null): string | null =>
  visualStudioPath
    ? path.join(
        visualStudioPath,
        'Common7',
        'IDE',
        'CommonExtensions',
        'Microsoft',
        'CMake',
        'CMake',
        'bin',
        'cmake.exe',
      )
    : null;

const bundledNinja = (visualStudioPath: string | null): string | null =>
  visualStudioPath
    ? path.join(visualStudioPath, 'Common7', 'IDE', 'CommonExtensions', 'Microsoft', 'CMake', 'Ninja', 'ninja.exe')
    : null;

const defaultCmakePaths = (environment: Environment): string[] =>
  [environment.ProgramFiles, environment['ProgramFiles(x86)']]
    .filter((value): value is string => Boolean(value))
    .map((value) => path.join(value, 'CMake', 'bin', 'cmake.exe'));

const firstValidOrFirst = (
  candidates: Candidate[],
  validate: (value: Candidate | null) => EditorPathValidation,
): Candidate | null => {
  for (const value of candidates) if (validate(value).valid) return value;
  return candidates[0] ?? null;
};

// Explicit overrides are authoritative so a bad manual path is visible instead of silently bypassed.
const configuredOrDetected = (
  configured: Candidate | null,
  detected: Candidate[],
  validate: (value: Candidate | null) => EditorPathValidation,
): Candidate | null => configured ?? firstValidOrFirst(detected, validate);

export const resolveWindowsToolchainValidation = (
  preferences: WindowsToolchainPreferences,
  environment: Environment = process.env,
): Record<string, EditorPathValidation> => {
  const configuredVisualStudio = candidate(preferences.visualStudioPath, 'configured');
  const visualStudioCandidate = configuredOrDetected(
    configuredVisualStudio,
    uniqueCandidates([
      candidate(environment.VSINSTALLDIR, 'environment'),
      candidate(visualStudioFromVcInstallDir(environment), 'environment'),
      candidate(visualStudioFromVswhere(environment), 'derived'),
      ...defaultVisualStudioRoots(environment).map((value) => candidate(value, 'default')),
    ]),
    validateVisualStudioRoot,
  );
  const visualStudioPath =
    visualStudioCandidate && validateVisualStudioRoot(visualStudioCandidate).valid ? visualStudioCandidate.path : null;

  const configuredMsvc = candidate(preferences.msvcToolchainPath, 'configured');
  const msvcCandidate = configuredOrDetected(
    configuredMsvc,
    uniqueCandidates([
      candidate(environment.VCToolsInstallDir, 'environment'),
      candidate(msvcUnderVisualStudio(visualStudioPath), 'derived'),
    ]),
    validateMsvcToolchainRoot,
  );

  const configuredSdk = candidate(preferences.sdkPath, 'configured');
  const sdkCandidate = configuredOrDetected(
    configuredSdk,
    uniqueCandidates([
      candidate(environment.WindowsSdkDir, 'environment'),
      ...defaultWindowsSdkRoots(environment).map((value) => candidate(value, 'default')),
    ]),
    validateWindowsSdkRoot,
  );

  const configuredCmake = candidate(preferences.cmakePath, 'configured');
  const cmakeCandidate = configuredOrDetected(
    configuredCmake,
    uniqueCandidates([
      candidate(environment.CMAKE_COMMAND, 'environment'),
      candidate(findExecutableOnPath('cmake', environment), 'environment'),
      candidate(bundledCmake(visualStudioPath), 'derived'),
      ...defaultCmakePaths(environment).map((value) => candidate(value, 'default')),
    ]),
    (value) => validateExecutable('CMake', value),
  );

  const configuredNinja = candidate(preferences.ninjaPath, 'configured');
  const ninjaCandidate = configuredOrDetected(
    configuredNinja,
    uniqueCandidates([
      candidate(environment.CMAKE_MAKE_PROGRAM, 'environment'),
      candidate(findExecutableOnPath('ninja', environment), 'environment'),
      candidate(bundledNinja(visualStudioPath), 'derived'),
    ]),
    (value) => validateExecutable('Ninja', value),
  );

  return {
    [windowsToolchainSettingKeys.visualStudioPath]: validateVisualStudioRoot(visualStudioCandidate),
    [windowsToolchainSettingKeys.msvcToolchainPath]: validateMsvcToolchainRoot(msvcCandidate),
    [windowsToolchainSettingKeys.sdkPath]: validateWindowsSdkRoot(sdkCandidate),
    [windowsToolchainSettingKeys.cmakePath]: validateExecutable('CMake', cmakeCandidate),
    [windowsToolchainSettingKeys.ninjaPath]: validateExecutable('Ninja', ninjaCandidate),
  };
};
