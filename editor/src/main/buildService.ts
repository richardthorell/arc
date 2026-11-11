import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

import type { ArcBuildDiagnostic, ArcBuildRequest, ArcBuildRun, ArcBuildSnapshot } from '../common/buildTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';
import type { WorkflowRequest, WorkflowPlan } from '../common/projectWorkflowTypes';
import { queryProjectWorkflowPlan } from './projectWorkflowPlanner';

type HostBridge = {
  command(
    type: string,
    payload?: Record<string, unknown>,
  ): Promise<{ succeeded: boolean; error?: string; payload?: unknown }>;
};

const diagnosticPattern =
  /^(.*?)(?:\((\d+)(?:,(\d+))?\)|:(\d+)(?::(\d+))?):\s*(?:(fatal error|error|warning)\s*(?:[A-Z]+\d+)?:?\s*)?(.*)$/i;

export class BuildService {
  private snapshotValue: ArcBuildSnapshot = {
    revision: 1,
    state: 'idle',
    configuration: 'RelWithDebInfo',
    buildRequired: false,
    reloadRequired: false,
    restartRequired: false,
    diagnostics: [],
    history: [],
  };
  private process: ChildProcessWithoutNullStreams | null = null;
  private sourceWatcher: fs.FSWatcher | null = null;
  private buildWatcher: fs.FSWatcher | null = null;
  private sequence = 0;
  private nextRunId = 0;
  private projectEpoch = 0;
  private cancelledProcesses = new WeakSet<ChildProcessWithoutNullStreams>();

  constructor(
    private readonly activeProject: () => ArcProjectCandidate | null,
    private readonly projectTool: () => string,
    private readonly host: HostBridge,
    private readonly publish: (snapshot: ArcBuildSnapshot) => void,
  ) {}

  snapshot(): ArcBuildSnapshot {
    return structuredClone(this.snapshotValue);
  }

  watchProject(): void {
    ++this.projectEpoch;
    this.sourceWatcher?.close();
    this.buildWatcher?.close();
    this.sourceWatcher = this.buildWatcher = null;
    const project = this.activeProject();
    this.update({
      history: [],
      state: 'idle',
      diagnostics: [],
      buildRequired: false,
      reloadRequired: false,
      restartRequired: false,
    });
    if (!project) return;
    const sourceRoot = path.join(project.projectRoot, project.descriptor.paths.source);
    if (fs.existsSync(sourceRoot)) {
      this.sourceWatcher = fs.watch(sourceRoot, { recursive: true }, (_event, file) => {
        if (!file || !/\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx)$/i.test(file)) return;
        this.update({ buildRequired: true, reloadRequired: false });
        this.append('info', `C++ changes detected: ${file}`, 'codegen');
      });
    }
    const editorState = path.join(project.projectRoot, project.descriptor.paths.saved, 'Editor');
    fs.mkdirSync(editorState, { recursive: true });
    this.buildWatcher = fs.watch(editorState, (_event, file) => {
      if (file === 'active-build.json' && !this.process) {
        this.update({ buildRequired: false, reloadRequired: true });
        this.append('info', 'External C++ build detected; module reload required', 'module');
      }
    });
  }

  async plan(request: WorkflowRequest): Promise<WorkflowPlan> {
    const project = this.activeProject();
    if (!project) throw new Error('No project is open');
    const epoch = this.projectEpoch;
    const descriptor = JSON.stringify(project.descriptor);
    const descriptorPath = project.descriptorPath;
    const tool = this.projectTool();
    const plan = await queryProjectWorkflowPlan(tool, descriptorPath, request);
    const current = this.activeProject();
    if (
      epoch !== this.projectEpoch ||
      current?.descriptorPath !== descriptorPath ||
      JSON.stringify(current?.descriptor) !== descriptor ||
      this.projectTool() !== tool ||
      plan.projectGuid !== current.descriptor.guid
    )
      throw new Error('Project or SDK changed while planning; request a new plan');
    return plan;
  }

  async execute(request: ArcBuildRequest): Promise<ArcBuildSnapshot> {
    if (request.action === 'cancel') {
      if (this.process) {
        this.cancelledProcesses.add(this.process);
        this.process.kill();
      }
      this.process = null;
      this.update({ state: 'cancelled', completedAt: new Date().toISOString() });
      return this.snapshot();
    }
    if (this.process) throw new Error('A project build is already running');
    const project = this.activeProject();
    if (!project) throw new Error('No project is open');
    if (!project.writable) throw new Error('Project build is disabled in read-only mode');
    const configuration = request.configuration ?? this.snapshotValue.configuration;
    if (configuration !== this.snapshotValue.configuration) this.update({ configuration });
    if (request.action === 'reload') {
      await this.reloadModule(project);
      return this.snapshot();
    }
    if (request.action === 'openIde') {
      await this.runTool(
        ['ide', 'generate', '--project', project.descriptorPath, '--ide', request.ide ?? 'vscode'],
        'configuring',
      );
      await this.runTool(
        ['ide', 'launch', '--project', project.descriptorPath, '--ide', request.ide ?? 'vscode'],
        'configuring',
      );
      return this.snapshot();
    }
    if (request.action === 'rebuild') {
      await this.runTool(
        ['build', '--project', project.descriptorPath, '--config', configuration, '--target', 'clean'],
        'cleaning',
      );
      await this.runTool(['configure', '--project', project.descriptorPath], 'configuring');
      await this.runTool(['build', '--project', project.descriptorPath, '--config', configuration], 'building');
      await this.reloadAfterBuild(project);
      return this.snapshot();
    }
    const arguments_ =
      request.action === 'configure'
        ? ['configure', '--project', project.descriptorPath]
        : request.action === 'clean'
          ? ['build', '--project', project.descriptorPath, '--config', configuration, '--target', 'clean']
          : ['build', '--project', project.descriptorPath, '--config', configuration];
    await this.runTool(
      arguments_,
      request.action === 'clean' ? 'cleaning' : request.action === 'configure' ? 'configuring' : 'building',
    );
    if (request.action === 'build') await this.reloadAfterBuild(project);
    return this.snapshot();
  }

  dispose(): void {
    this.process?.kill();
    this.sourceWatcher?.close();
    this.buildWatcher?.close();
  }

  private async runTool(arguments_: string[], state: ArcBuildSnapshot['state']): Promise<void> {
    const tool = this.projectTool();
    if (!tool || !fs.existsSync(tool)) throw new Error('ARC project build tool is unavailable');
    const project = this.activeProject();
    if (!project) throw new Error('No project is open');
    const effective = arguments_;
    const startedMs = Date.now();
    const kind: ArcBuildRun['kind'] =
      effective[0] === 'ide' ? 'ide' : state === 'cleaning' ? 'clean' : state === 'configuring' ? 'configure' : 'build';
    const configuration = this.snapshotValue.configuration;
    this.snapshotValue.diagnostics = [];
    this.update({
      state,
      startedAt: new Date().toISOString(),
      completedAt: undefined,
      command: `${tool} ${effective.join(' ')}`,
    });
    await new Promise<void>((resolve, reject) => {
      const child = spawn(tool, effective, { cwd: project.projectRoot, shell: false, windowsHide: true });
      this.process = child;
      let failedToStart = false;
      let pending = '';
      const consume = (data: Buffer, severity: ArcBuildDiagnostic['severity']) => {
        pending += data.toString('utf8');
        const lines = pending.split(/\r?\n/);
        pending = lines.pop() ?? '';
        for (const line of lines) if (line.trim()) this.parseLine(line, severity);
      };
      child.stdout.on('data', (data: Buffer) => consume(data, 'info'));
      child.stderr.on('data', (data: Buffer) => consume(data, 'error'));
      child.on('error', (error) => {
        failedToStart = true;
        this.process = null;
        this.append('error', error.message, 'compiler');
        this.recordRun(kind, configuration, startedMs, 'failed');
        this.update({ state: 'failed', buildRequired: true, completedAt: new Date().toISOString() });
        reject(error);
      });
      child.on('close', (code) => {
        if (failedToStart) return;
        if (pending.trim()) this.parseLine(pending, code === 0 ? 'info' : 'error');
        this.process = null;
        if (this.cancelledProcesses.delete(child)) {
          this.recordRun(kind, configuration, startedMs, 'cancelled');
          this.update({ state: 'cancelled', completedAt: new Date().toISOString() });
          reject(new Error('Project build cancelled'));
          return;
        }
        this.recordRun(kind, configuration, startedMs, code === 0 ? 'succeeded' : 'failed');
        if (code === 0) {
          this.update({ state: 'succeeded', buildRequired: false, completedAt: new Date().toISOString() });
          resolve();
        } else {
          this.update({ state: 'failed', buildRequired: true, completedAt: new Date().toISOString() });
          reject(new Error(`Project build exited with code ${String(code)}`));
        }
      });
    });
  }

  private recordRun(
    kind: ArcBuildRun['kind'],
    configuration: string,
    startedMs: number,
    result: ArcBuildRun['result'],
  ): void {
    const diagnostics = this.snapshotValue.diagnostics;
    const run: ArcBuildRun = {
      id: ++this.nextRunId,
      kind,
      configuration,
      startedAt: new Date(startedMs).toISOString(),
      durationMs: Math.max(0, Date.now() - startedMs),
      result,
      warningCount: diagnostics.filter((item) => item.severity === 'warning').length,
      errorCount: diagnostics.filter((item) => item.severity === 'error').length,
    };
    this.update({ history: [...(this.snapshotValue.history ?? []).slice(-11), run] });
  }

  private async reloadAfterBuild(project: ArcProjectCandidate): Promise<void> {
    try {
      await this.reloadModule(project);
    } catch (error) {
      this.update({ state: 'failed', reloadRequired: true, completedAt: new Date().toISOString() });
      throw error;
    }
  }

  private async reloadModule(project: ArcProjectCandidate): Promise<void> {
    const activeBuildPath = path.join(
      project.projectRoot,
      project.descriptor.paths.saved,
      'Editor',
      'active-build.json',
    );
    const activeBuild = JSON.parse(fs.readFileSync(activeBuildPath, 'utf8')) as { moduleManifest?: string };
    if (!activeBuild.moduleManifest) throw new Error('Build completed without a module manifest');
    const manifest = JSON.parse(
      fs.readFileSync(path.resolve(project.projectRoot, activeBuild.moduleManifest), 'utf8'),
    ) as {
      modules?: Array<{ id?: string; path?: string }>;
    };
    const moduleId = project.descriptor.modules.find((module) => module.kind === 'editor' && module.enabled)?.id;
    const module = manifest.modules?.find((candidate) => candidate.id === moduleId);
    if (!moduleId || !module?.path) throw new Error('Build manifest does not contain the project editor module');
    const response = await this.host.command('project.reloadModule', {
      path: module.path,
      engineVersion: project.descriptor.engineVersion,
      projectGuid: project.descriptor.guid,
      moduleId,
    });
    if (!response.succeeded) {
      this.update({ reloadRequired: true });
      this.append('error', response.error ?? 'Project module reload failed', 'module');
      throw new Error(response.error ?? 'Project module reload failed');
    }
    const payload = response.payload as { classification?: string } | undefined;
    const restartRequired = payload?.classification === 'nativeHostRestartRequired';
    this.update({
      reloadRequired: restartRequired,
      restartRequired,
    });
    this.append('info', `Module reload: ${payload?.classification ?? 'complete'}`, 'module');
  }

  private parseLine(line: string, fallback: ArcBuildDiagnostic['severity']): void {
    const match = diagnosticPattern.exec(line);
    const severity = /warning/i.test(match?.[6] ?? '')
      ? 'warning'
      : /error/i.test(match?.[6] ?? '')
        ? 'error'
        : fallback;
    this.snapshotValue.diagnostics.push({
      sequence: ++this.sequence,
      severity,
      message: match?.[7]?.trim() || line.trim(),
      file: match?.[1] && /[\\/]|\.[a-z]+$/i.test(match[1]) ? match[1] : undefined,
      line: Number(match?.[2] ?? match?.[4]) || undefined,
      column: Number(match?.[3] ?? match?.[5]) || undefined,
      category: /link|LNK\d+/i.test(line) ? 'linker' : /reflection|codegen/i.test(line) ? 'codegen' : 'compiler',
    });
    this.publish(this.snapshot());
  }

  private append(
    severity: ArcBuildDiagnostic['severity'],
    message: string,
    category: ArcBuildDiagnostic['category'],
  ): void {
    this.snapshotValue.diagnostics.push({ sequence: ++this.sequence, severity, message, category });
    this.publish(this.snapshot());
  }

  private update(changes: Partial<ArcBuildSnapshot>): void {
    this.snapshotValue = { ...this.snapshotValue, ...changes, revision: this.snapshotValue.revision + 1 };
    this.publish(this.snapshot());
  }
}
