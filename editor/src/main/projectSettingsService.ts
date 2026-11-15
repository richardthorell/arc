import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import type { ArcProjectCandidate } from '../common/projectTypes';
import {
  projectSettingsCatalog,
  projectSettingsCapabilitiesSchema,
  projectSettingValue,
  validateProjectSettingChanges,
  type ProjectSettingsSnapshot,
  type ProjectSettingsApplyRequest,
  type ProjectSettingsApplyResult,
} from '../common/projectSettingsTypes';

const executeFile = promisify(execFile);
async function nativeQuery(tool: string, args: string[]): Promise<unknown> {
  try {
    const result = await executeFile(tool, [...args, '--json'], {
      windowsHide: true,
      timeout: 30_000,
      maxBuffer: 1024 * 1024,
    });
    return JSON.parse(result.stdout);
  } catch (error) {
    const stdout = (error as { stdout?: string }).stdout;
    if (stdout) {
      try {
        const value = JSON.parse(stdout);
        if (value.error) throw new Error(value.error);
      } catch (parsed) {
        if (parsed instanceof SyntaxError) throw error;
        throw parsed;
      }
    }
    throw error;
  }
}

function readDescriptor(file: string): { raw: Record<string, unknown>; bytes: string } {
  if (fs.statSync(file).size > 1024 * 1024) throw new Error('Project descriptor exceeds size limit');
  const bytes = fs.readFileSync(file, 'utf8');
  const raw = JSON.parse(bytes);
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Invalid project descriptor');
  return { raw, bytes };
}

/** Only catalog IDs can address storage. Preserve unknown keys in edited collection members. */
function mergeValue(previous: unknown, next: unknown): unknown {
  if (Array.isArray(next))
    return next.map((item) => {
      const identity = item && typeof item === 'object' ? (item.id ?? item.guid) : undefined;
      const old =
        identity && Array.isArray(previous)
          ? previous.find((value) => (value.id ?? value.guid) === identity)
          : undefined;
      return mergeValue(old, item);
    });
  if (next && typeof next === 'object') {
    const result: Record<string, unknown> =
      previous && typeof previous === 'object' && !Array.isArray(previous) ? { ...previous } : {};
    for (const [key, value] of Object.entries(next)) {
      if (['__proto__', 'prototype', 'constructor'].includes(key)) throw new Error('Unsafe settings key');
      result[key] = mergeValue(result[key], value);
    }
    return result;
  }
  return next;
}

export class ProjectSettingsService {
  private applying = false;
  private epoch = 0;
  private observed: ArcProjectCandidate | null = null;
  private capabilities?: { tool: string; promise: Promise<ProjectSettingsSnapshot['capabilities']> };
  private resolvedCapabilities?: ProjectSettingsSnapshot['capabilities'];
  private capabilitiesError?: string;
  constructor(
    private readonly active: () => ArcProjectCandidate | null,
    private readonly tool: () => string,
    private readonly workflowBusy: () => boolean,
    private readonly publish: (raw: Record<string, unknown>, expected: ArcProjectCandidate) => void,
    private readonly query = nativeQuery,
  ) {}
  isBusy(): boolean {
    return this.applying;
  }
  describe() {
    return structuredClone(projectSettingsCatalog);
  }
  private revision(project: ArcProjectCandidate, bytes: string): string {
    if (this.observed !== project) {
      this.observed = project;
      ++this.epoch;
    }
    return createHash('sha256').update(`${this.epoch}\n${project.descriptorPath}\n${bytes}`).digest('hex');
  }
  async read(): Promise<ProjectSettingsSnapshot> {
    const project = this.active();
    if (!project) throw new Error('Open a project to edit Project Settings');
    const tool = this.tool();
    if (!this.capabilities || this.capabilities.tool !== tool) {
      this.resolvedCapabilities = undefined;
      this.capabilitiesError = undefined;
      this.capabilities = {
        tool,
        promise: this.query(tool, ['settings', 'describe']).then((value) => {
          const parsed = projectSettingsCapabilitiesSchema.parse(value);
          if (this.capabilities?.tool === tool) this.resolvedCapabilities = parsed;
          return parsed;
        }),
      };
      void this.capabilities.promise.catch(() => {
        if (this.capabilities?.tool === tool)
          this.capabilitiesError = 'Native target options are unavailable. Check the configured project tool.';
      });
    }
    if (project !== this.active()) throw new Error('Project changed while reading settings');
    return {
      ...this.snapshotFor(
        project,
        this.resolvedCapabilities ?? {
          version: 1,
          targets: [],
          textureOutputs: [],
          textureQualities: [],
          androidMinimumSdk: 26,
          androidCompileSdk: 35,
        },
      ),
      capabilitiesPending: !this.resolvedCapabilities && !this.capabilitiesError,
      capabilitiesError: this.capabilitiesError,
    };
  }
  private snapshotFor(
    project: ArcProjectCandidate,
    capabilities: ProjectSettingsSnapshot['capabilities'],
  ): ProjectSettingsSnapshot {
    const { raw, bytes } = readDescriptor(project.descriptorPath);
    if (raw.guid !== project.descriptor.guid)
      throw new Error('Project identity changed externally; reopen the project');
    const values: Record<string, unknown> = {};
    for (const field of projectSettingsCatalog.fields)
      values[field.id] =
        field.id === 'descriptorPath'
          ? project.descriptorPath
          : (projectSettingValue(raw, field.id) ?? ('defaultValue' in field ? field.defaultValue : ''));
    return {
      catalog: this.describe(),
      projectGuid: project.descriptor.guid,
      revision: this.revision(project, bytes),
      writable: project.writable,
      values,
      capabilities,
      configurations: Array.isArray(raw.buildConfigurations)
        ? raw.buildConfigurations.filter((entry): entry is string => typeof entry === 'string')
        : project.descriptor.buildConfigurations,
    };
  }
  async apply(request: ProjectSettingsApplyRequest): Promise<ProjectSettingsApplyResult> {
    if (this.applying) return { succeeded: false, errors: { _form: 'Project Settings are already being applied' } };
    this.applying = true;
    let temporary: string | undefined;
    try {
      if (
        !request ||
        request.catalogVersion !== 1 ||
        !request.changes ||
        typeof request.changes !== 'object' ||
        Array.isArray(request.changes) ||
        JSON.stringify(request).length > 256 * 1024
      )
        throw new Error('Unsupported or invalid Project Settings request');
      const errors = validateProjectSettingChanges(request.changes);
      if (Object.keys(errors).length) return { succeeded: false, errors };
      const project = this.active();
      if (!project?.writable) throw new Error('Project is unavailable or read-only');
      if (fs.lstatSync(project.descriptorPath).isSymbolicLink())
        throw new Error('Editing a linked project descriptor is not supported');
      const tool = this.tool();
      const metadata = await this.read();
      metadata.capabilities = await this.capabilities!.promise;
      const { raw, bytes } = readDescriptor(project.descriptorPath);
      const assertCurrent = () => {
        if (
          this.active() !== project ||
          this.tool() !== tool ||
          !project.writable ||
          request.projectGuid !== project.descriptor.guid ||
          request.revision !== this.revision(project, fs.readFileSync(project.descriptorPath, 'utf8'))
        )
          throw new Error('Project Settings changed externally; reload before applying');
        if (this.workflowBusy()) throw new Error('Wait for the active build/cook workflow before applying settings');
      };
      assertCurrent();
      const next = JSON.parse(bytes) as Record<string, unknown>;
      for (const [id, value] of Object.entries(request.changes)) {
        const keys = id.split('.');
        let object = next;
        for (const key of keys.slice(0, -1)) {
          if (!object[key] || typeof object[key] !== 'object' || Array.isArray(object[key])) object[key] = {};
          object = object[key] as Record<string, unknown>;
        }
        const key = keys.at(-1)!;
        object[key] = mergeValue(object[key], value);
      }
      if (next.guid !== raw.guid) throw new Error('Project identity cannot change');
      // Same parent ensures native relative-path validation uses the real project root.
      temporary = path.join(project.projectRoot, `.arc-settings-${randomUUID()}.arcproject`);
      fs.writeFileSync(temporary, JSON.stringify(next, null, 2) + '\n', { flag: 'wx' });
      const validated = await this.query(tool, ['validate', '--project', temporary]);
      if (!(validated as { succeeded?: boolean })?.succeeded) throw new Error('Native project validation failed');
      assertCurrent();
      // No async boundary between final authority check and publication.
      fs.renameSync(temporary, project.descriptorPath);
      temporary = undefined;
      this.publish(next, project);
      return { succeeded: true, snapshot: this.snapshotFor(this.active()!, metadata.capabilities) };
    } catch (error) {
      return { succeeded: false, errors: { _form: error instanceof Error ? error.message : String(error) } };
    } finally {
      if (temporary) fs.rmSync(temporary, { force: true });
      this.applying = false;
    }
  }
}
