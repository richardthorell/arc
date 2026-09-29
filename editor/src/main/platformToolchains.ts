import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import type { EditorHostPlatform, EditorPathValidation } from '../common/editorWorkflowTypes';

export const buildToolSettingKeys = {
  cmakePath: 'platform.windows.cmakePath',
  ninjaPath: 'platform.windows.ninjaPath',
} as const;

export const linuxToolchainSettingKeys = {
  buildEnvironment: 'platform.linux.buildEnvironment',
  wslDistribution: 'platform.linux.wslDistribution',
  compilerPath: 'platform.linux.compilerPath',
  sysrootPath: 'platform.linux.sysrootPath',
} as const;

export const appleToolchainSettingKeys = {
  xcodePath: 'platform.apple.xcodePath',
  macosSdkPath: 'platform.apple.macosSdkPath',
  iosDeviceSdkPath: 'platform.apple.iosDeviceSdkPath',
  iosSimulatorSdkPath: 'platform.apple.iosSimulatorSdkPath',
  tvosDeviceSdkPath: 'platform.apple.tvosDeviceSdkPath',
  tvosSimulatorSdkPath: 'platform.apple.tvosSimulatorSdkPath',
  visionosDeviceSdkPath: 'platform.apple.visionosDeviceSdkPath',
  visionosSimulatorSdkPath: 'platform.apple.visionosSimulatorSdkPath',
} as const;

export const webToolchainSettingKeys = {
  emsdkPath: 'platform.web.emsdkPath',
} as const;

export const licensedPlatformSettingKeys = {
  xboxGdkPath: 'platform.xbox.gdkPath',
  playStationSdkPath: 'platform.playstation.sdkPath',
  switchSdkPath: 'platform.switch.sdkPath',
} as const;

type Environment = Record<string, string | undefined>;
type CommandRunner = (executable: string, arguments_: string[], environment?: Environment) => string;

type Candidate = {
  path: string;
  source: EditorPathValidation['source'];
};

const defaultRunner: CommandRunner = (executable, arguments_, environment) =>
  execFileSync(executable, arguments_, {
    encoding: 'utf8',
    windowsHide: true,
    env: environment ? { ...process.env, ...environment } : process.env,
  }).trim();

export const editorHostPlatform = (platform: NodeJS.Platform = process.platform): EditorHostPlatform => {
  if (platform === 'win32') return 'windows';
  if (platform === 'darwin') return 'macos';
  if (platform === 'linux') return 'linux';
  return 'other';
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

const executableNames = (name: string, platform: NodeJS.Platform): string[] =>
  platform === 'win32' ? [`${name}.exe`, name] : [name, `${name}.exe`];

const findExecutableOnPath = (name: string, environment: Environment, platform: NodeJS.Platform): string | null => {
  const pathValue = environment.PATH ?? environment.Path ?? environment.path;
  if (!pathValue) return null;
  for (const entry of pathValue.split(path.delimiter)) {
    if (!entry) continue;
    for (const executable of executableNames(name, platform)) {
      const executablePath = path.join(entry, executable);
      if (fileExists(executablePath)) return executablePath;
    }
  }
  return null;
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
  return { valid: true, resolvedPath: value.path, message: `Validated · ${label}`, source: value.source };
};

export const resolveBuildToolValidation = (
  preferences: { cmakePath?: string; ninjaPath?: string },
  environment: Environment = process.env,
  platform: NodeJS.Platform = process.platform,
): Record<string, EditorPathValidation> => {
  const cmake =
    candidate(preferences.cmakePath, 'configured') ??
    candidate(environment.CMAKE_COMMAND, 'environment') ??
    candidate(findExecutableOnPath('cmake', environment, platform), 'environment');
  const ninja =
    candidate(preferences.ninjaPath, 'configured') ??
    candidate(environment.CMAKE_MAKE_PROGRAM, 'environment') ??
    candidate(findExecutableOnPath('ninja', environment, platform), 'environment');
  return {
    [buildToolSettingKeys.cmakePath]: validateExecutable('CMake', cmake),
    [buildToolSettingKeys.ninjaPath]: validateExecutable('Ninja', ninja),
  };
};

const unsupported = (message: string): EditorPathValidation => ({
  valid: false,
  resolvedPath: '',
  message,
  source: 'unresolved',
});

const runOptional = (
  runner: CommandRunner,
  executable: string,
  arguments_: string[],
  environment?: Environment,
): string => {
  try {
    return runner(executable, arguments_, environment).replaceAll('\0', '').trim();
  } catch {
    return '';
  }
};

const validateDirectory = (label: string, value: Candidate | null): EditorPathValidation => {
  if (!value) return { valid: false, resolvedPath: '', message: `${label} not detected`, source: 'unresolved' };
  if (!directoryExists(value.path)) {
    return { valid: false, resolvedPath: value.path, message: `Path is not a valid ${label}`, source: value.source };
  }
  return { valid: true, resolvedPath: value.path, message: `Validated · ${label}`, source: value.source };
};

export const resolveLinuxToolchainValidation = (
  preferences: {
    buildEnvironment?: string;
    wslDistribution?: string;
    compilerPath?: string;
    sysrootPath?: string;
  },
  environment: Environment = process.env,
  platform: NodeJS.Platform = process.platform,
  runner: CommandRunner = defaultRunner,
): Record<string, EditorPathValidation> => {
  const mode = preferences.buildEnvironment || 'auto';
  if (platform === 'linux') {
    if (mode === 'wsl') {
      const unavailable = unsupported('WSL is only available from a Windows host');
      return {
        [linuxToolchainSettingKeys.compilerPath]: unavailable,
        [linuxToolchainSettingKeys.sysrootPath]: unavailable,
      };
    }
    const compiler =
      candidate(preferences.compilerPath, 'configured') ??
      candidate(findExecutableOnPath('clang++', environment, platform), 'environment') ??
      candidate(findExecutableOnPath('g++', environment, platform), 'environment');
    const sysroot = candidate(preferences.sysrootPath, 'configured') ?? candidate('/', 'default');
    return {
      [linuxToolchainSettingKeys.compilerPath]: validateExecutable('Linux C++ compiler', compiler),
      [linuxToolchainSettingKeys.sysrootPath]: validateDirectory('Linux sysroot', sysroot),
    };
  }

  if (platform !== 'win32' || mode === 'local') {
    const unavailable = unsupported(
      platform === 'win32'
        ? 'Local Linux builds require a Linux host; select WSL instead'
        : 'Local Linux builds are unavailable on this host; use a remote build host in a future ARC release',
    );
    return {
      [linuxToolchainSettingKeys.compilerPath]: unavailable,
      [linuxToolchainSettingKeys.sysrootPath]: unavailable,
    };
  }

  const wslExecutable =
    findExecutableOnPath('wsl', environment, platform) ??
    (environment.SystemRoot ? path.join(environment.SystemRoot, 'System32', 'wsl.exe') : 'wsl.exe');
  const listed = runOptional(runner, wslExecutable, ['-l', '-q']);
  const distribution =
    preferences.wslDistribution?.trim() ||
    listed
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find(Boolean) ||
    '';
  if (!distribution) {
    const unavailable = unsupported('WSL is not configured');
    return {
      [linuxToolchainSettingKeys.compilerPath]: unavailable,
      [linuxToolchainSettingKeys.sysrootPath]: unavailable,
    };
  }

  const detectedCompiler = runOptional(runner, wslExecutable, [
    '-d',
    distribution,
    '--',
    'sh',
    '-lc',
    'command -v clang++ || command -v g++',
  ]);
  const compilerPath = preferences.compilerPath?.trim() || detectedCompiler;
  const compilerValid = Boolean(
    compilerPath &&
    runOptional(runner, wslExecutable, ['-d', distribution, '--', 'sh', '-lc', `test -x '${compilerPath}' && echo ok`]),
  );
  const sysrootPath = preferences.sysrootPath?.trim() || '/';
  const sysrootValid = Boolean(
    runOptional(runner, wslExecutable, ['-d', distribution, '--', 'sh', '-lc', `test -d '${sysrootPath}' && echo ok`]),
  );

  return {
    [linuxToolchainSettingKeys.compilerPath]: compilerValid
      ? {
          valid: true,
          resolvedPath: compilerPath,
          message: `Validated · WSL ${distribution} C++ compiler`,
          source: preferences.compilerPath ? 'configured' : 'derived',
        }
      : unsupported(`No usable C++ compiler was found in WSL distribution '${distribution}'`),
    [linuxToolchainSettingKeys.sysrootPath]: sysrootValid
      ? {
          valid: true,
          resolvedPath: sysrootPath,
          message: `Validated · WSL ${distribution} sysroot`,
          source: preferences.sysrootPath ? 'configured' : 'derived',
        }
      : unsupported(`Linux sysroot '${sysrootPath}' is not available in WSL distribution '${distribution}'`),
  };
};

const appleSdkNames: ReadonlyArray<[key: string, sdk: string, label: string]> = [
  [appleToolchainSettingKeys.macosSdkPath, 'macosx', 'macOS SDK'],
  [appleToolchainSettingKeys.iosDeviceSdkPath, 'iphoneos', 'iOS Device SDK'],
  [appleToolchainSettingKeys.iosSimulatorSdkPath, 'iphonesimulator', 'iOS Simulator SDK'],
  [appleToolchainSettingKeys.tvosDeviceSdkPath, 'appletvos', 'tvOS Device SDK'],
  [appleToolchainSettingKeys.tvosSimulatorSdkPath, 'appletvsimulator', 'tvOS Simulator SDK'],
  [appleToolchainSettingKeys.visionosDeviceSdkPath, 'xros', 'visionOS Device SDK'],
  [appleToolchainSettingKeys.visionosSimulatorSdkPath, 'xrsimulator', 'visionOS Simulator SDK'],
];

export const resolveAppleToolchainValidation = (
  preferences: { xcodePath?: string },
  environment: Environment = process.env,
  platform: NodeJS.Platform = process.platform,
  runner: CommandRunner = defaultRunner,
): Record<string, EditorPathValidation> => {
  if (platform !== 'darwin') {
    const result: Record<string, EditorPathValidation> = {
      [appleToolchainSettingKeys.xcodePath]: unsupported('Local Apple builds require macOS and Xcode'),
    };
    for (const [key] of appleSdkNames) result[key] = unsupported('SDK discovery requires macOS and Xcode');
    return result;
  }

  const detectedXcode = runOptional(runner, '/usr/bin/xcode-select', ['-p']);
  const xcode =
    candidate(preferences.xcodePath, 'configured') ??
    candidate(environment.DEVELOPER_DIR, 'environment') ??
    candidate(detectedXcode, 'derived');
  const xcodeValidation = validateDirectory('Xcode developer directory', xcode);
  const result: Record<string, EditorPathValidation> = {
    [appleToolchainSettingKeys.xcodePath]: xcodeValidation,
  };
  for (const [key, sdk, label] of appleSdkNames) {
    if (!xcodeValidation.valid) {
      result[key] = unsupported(`${label} requires a valid Xcode installation`);
      continue;
    }
    const sdkPath = runOptional(runner, '/usr/bin/xcrun', ['--sdk', sdk, '--show-sdk-path'], {
      ...environment,
      DEVELOPER_DIR: xcodeValidation.resolvedPath,
    });
    result[key] =
      sdkPath && directoryExists(sdkPath)
        ? { valid: true, resolvedPath: sdkPath, message: `Validated · ${label}`, source: 'derived' }
        : unsupported(`${label} is not installed in the selected Xcode installation`);
  }
  return result;
};

export const resolveWebToolchainValidation = (
  preferences: { emsdkPath?: string },
  environment: Environment = process.env,
): Record<string, EditorPathValidation> => {
  const root = candidate(preferences.emsdkPath, 'configured') ?? candidate(environment.EMSDK, 'environment');
  if (!root)
    return {
      [webToolchainSettingKeys.emsdkPath]: unsupported('Emscripten SDK not detected; set EMSDK or choose an SDK root'),
    };
  const emcc = [
    path.join(root.path, 'upstream', 'emscripten', 'emcc'),
    path.join(root.path, 'upstream', 'emscripten', 'emcc.bat'),
  ].find(fileExists);
  return {
    [webToolchainSettingKeys.emsdkPath]: emcc
      ? { valid: true, resolvedPath: root.path, message: 'Validated · Emscripten SDK', source: root.source }
      : {
          valid: false,
          resolvedPath: root.path,
          message: 'SDK root is missing upstream/emscripten/emcc',
          source: root.source,
        },
  };
};

export const resolveLicensedPlatformValidation = (
  preferences: { xboxGdkPath?: string; playStationSdkPath?: string; switchSdkPath?: string },
  platform: NodeJS.Platform = process.platform,
): Record<string, EditorPathValidation> => {
  const validate = (label: string, configured: string | undefined): EditorPathValidation => {
    if (platform !== 'win32') return unsupported(`${label} local builds require a supported Windows development host`);
    const root = candidate(configured, 'configured');
    if (!root) return unsupported(`${label} not configured`);
    return validateDirectory(label, root);
  };
  return {
    [licensedPlatformSettingKeys.xboxGdkPath]: validate('Microsoft GDK', preferences.xboxGdkPath),
    [licensedPlatformSettingKeys.playStationSdkPath]: validate('PlayStation SDK', preferences.playStationSdkPath),
    [licensedPlatformSettingKeys.switchSdkPath]: validate('Nintendo SDK', preferences.switchSdkPath),
  };
};

export const defaultLinuxBuildEnvironment = (
  platform: NodeJS.Platform = process.platform,
): 'auto' | 'local' | 'wsl' => {
  if (platform === 'linux') return 'local';
  if (platform === 'win32') return 'wsl';
  return 'auto';
};

export const defaultEmsdkRoots = (): string[] => [path.join(os.homedir(), 'emsdk')];
