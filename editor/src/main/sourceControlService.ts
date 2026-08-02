import { spawn } from 'node:child_process';
import path from 'node:path';

import type {
  SourceControlFile,
  SourceControlFileState,
  SourceControlResult,
  SourceControlSnapshot,
} from '../common/editorWorkflowTypes';

const statusCode = (value: string): SourceControlFileState | null => {
  switch (value) {
    case 'M':
      return 'modified';
    case 'A':
      return 'added';
    case 'D':
      return 'deleted';
    case 'R':
      return 'renamed';
    case 'C':
      return 'copied';
    case '?':
      return 'untracked';
    case 'U':
      return 'conflicted';
    default:
      return value === ' ' || value === '.' ? null : 'conflicted';
  }
};

const normalizedRepositoryPath = (value: string): string => value.replaceAll('\\', '/').replace(/^\/+/, '');

export class SourceControlService {
  constructor(
    private readonly projectRoot: () => string | null,
    private readonly enabled: () => boolean = () => true,
    private readonly writable: () => boolean = () => true,
  ) {}

  async snapshot(): Promise<SourceControlSnapshot> {
    const root = this.projectRoot();
    if (!root) return this.unavailable('No project is open');
    const status = await this.git(['status', '--porcelain=v2', '--branch', '-z']);
    if (!status.succeeded) return this.unavailable(status.error);
    const entries = status.output.split('\0').filter(Boolean);
    let branch = '';
    let detached = false;
    let ahead = 0;
    let behind = 0;
    const files: SourceControlFile[] = [];
    for (let index = 0; index < entries.length; ++index) {
      const entry = entries[index];
      if (entry.startsWith('# branch.head ')) {
        branch = entry.slice('# branch.head '.length);
        detached = branch === '(detached)';
      } else if (entry.startsWith('# branch.ab ')) {
        const match = entry.match(/\+(\d+)\s+-(\d+)/);
        ahead = Number(match?.[1] ?? 0);
        behind = Number(match?.[2] ?? 0);
      } else if (entry.startsWith('? ')) {
        files.push({ path: normalizedRepositoryPath(entry.slice(2)), indexState: null, worktreeState: 'untracked' });
      } else if (entry.startsWith('u ')) {
        const fields = entry.split(' ');
        files.push({
          path: normalizedRepositoryPath(fields.slice(10).join(' ')),
          indexState: 'conflicted',
          worktreeState: 'conflicted',
        });
      } else if (entry.startsWith('1 ') || entry.startsWith('2 ')) {
        const fields = entry.split(' ');
        const xy = fields[1] || '..';
        const filePath = normalizedRepositoryPath(fields.slice(entry.startsWith('2 ') ? 9 : 8).join(' '));
        const file: SourceControlFile = {
          path: filePath,
          indexState: statusCode(xy[0]),
          worktreeState: statusCode(xy[1]),
        };
        if (entry.startsWith('2 ') && entries[index + 1])
          file.originalPath = normalizedRepositoryPath(entries[++index]);
        files.push(file);
      }
    }
    files.sort((left, right) => left.path.localeCompare(right.path));
    return {
      available: true,
      repositoryRoot: root,
      branch,
      detached,
      ahead,
      behind,
      files,
      error: '',
    };
  }

  diff(filePath: string, staged = false): Promise<SourceControlResult> {
    return this.git(['diff', ...(staged ? ['--cached'] : []), '--', this.safePath(filePath)]);
  }

  stage(paths: string[]): Promise<SourceControlResult> {
    return this.mutate(['add', '--', ...paths.map((entry) => this.safePath(entry))]);
  }

  unstage(paths: string[]): Promise<SourceControlResult> {
    return this.mutate(['restore', '--staged', '--', ...paths.map((entry) => this.safePath(entry))]);
  }

  discard(paths: string[]): Promise<SourceControlResult> {
    return this.mutate(['restore', '--worktree', '--', ...paths.map((entry) => this.safePath(entry))]);
  }

  checkout(reference: string): Promise<SourceControlResult> {
    if (!/^[a-zA-Z0-9._/-]+$/.test(reference)) return Promise.resolve(this.failed('Invalid branch or reference'));
    return this.mutate(['switch', '--', reference]);
  }

  pull(): Promise<SourceControlResult> {
    return this.mutate(['pull', '--ff-only']);
  }

  push(): Promise<SourceControlResult> {
    return this.mutate(['push']);
  }

  commit(message: string): Promise<SourceControlResult> {
    if (!message.trim()) return Promise.resolve(this.failed('Commit message is required'));
    return this.mutate(['commit', '-m', message.trim()]);
  }

  private safePath(value: string): string {
    const normalized = normalizedRepositoryPath(value);
    if (!normalized || normalized === '..' || normalized.startsWith('../') || path.isAbsolute(normalized))
      throw new Error('Source-control path must remain inside the project');
    return normalized;
  }

  private async git(args: string[]): Promise<SourceControlResult> {
    if (!this.enabled()) return this.failed('Source control is disabled in editor settings');
    const root = this.projectRoot();
    if (!root) return this.failed('No project is open');
    return new Promise((resolve) => {
      const child = spawn('git', ['-C', root, ...args], {
        shell: false,
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let output = '';
      let error = '';
      child.stdout.on('data', (chunk) => {
        output += String(chunk);
      });
      child.stderr.on('data', (chunk) => {
        error += String(chunk);
      });
      child.once('error', (reason) => resolve(this.failed(reason.message)));
      child.once('exit', (code) =>
        resolve({
          succeeded: code === 0,
          output,
          error: code === 0 ? '' : error.trim() || `git exited with code ${String(code)}`,
        }),
      );
    });
  }

  private mutate(args: string[]): Promise<SourceControlResult> {
    if (!this.writable()) return Promise.resolve(this.failed('The active project is read-only'));
    return this.git(args);
  }

  private unavailable(error: string): SourceControlSnapshot {
    return {
      available: false,
      repositoryRoot: this.projectRoot() ?? '',
      branch: '',
      detached: false,
      ahead: 0,
      behind: 0,
      files: [],
      error,
    };
  }

  private failed(error: string): SourceControlResult {
    return { succeeded: false, output: '', error };
  }
}
