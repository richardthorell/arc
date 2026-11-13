import { execFile, spawn } from 'node:child_process';
import path from 'node:path';

export type WorkflowProcessRequest = {
  executable: string;
  args: string[];
  cwd: string;
  signal: AbortSignal;
  onOutput?: (text: string) => void;
  timeoutMs?: number;
  /** Text-only compiler logs keep a bounded tail; protocol responses must remain complete. */
  outputMode?: 'protocol' | 'log';
};

/** Bounded output and lifetime; cancellation addresses only the owned process tree. */
export async function runWorkflowProcess(request: WorkflowProcessRequest): Promise<string> {
  request.signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const child = spawn(request.executable, request.args, {
      cwd: request.cwd,
      shell: false,
      windowsHide: true,
      detached: process.platform !== 'win32',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    let failure: Error | undefined;
    let stopping: Promise<void> | undefined;
    let closed = false;
    const stop = (message: string) => {
      if (closed || stopping) return;
      failure = new Error(message);
      stopping = new Promise<void>((done) => {
        if (!child.pid) {
          done();
          return;
        }
        if (process.platform === 'win32') {
          execFile(
            path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'taskkill.exe'),
            ['/PID', String(child.pid), '/T', '/F'],
            { windowsHide: true, shell: false, timeout: 10_000 },
            (error) => {
              if (error && !closed) {
                failure = new Error(`${message}; process-tree termination failed: ${error.message}`);
                child.kill();
              }
              done();
            },
          );
        } else {
          try {
            process.kill(-child.pid, 'SIGKILL');
          } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== 'ESRCH')
              failure = new Error(`${message}; termination failed: ${String(error)}`);
          }
          done();
        }
      });
    };
    const abort = () => stop('Workflow cancelled');
    request.signal.addEventListener('abort', abort, { once: true });
    const timeout = setTimeout(() => stop('Workflow command timed out'), request.timeoutMs ?? 30 * 60_000);
    const consume = (data: Buffer, isError: boolean) => {
      const text = data.toString('utf8');
      if (isError) stderr = (stderr + text).slice(-64 * 1024);
      else {
        stdout += text;
        if (stdout.length > 1024 * 1024) {
          stdout = stdout.slice(-1024 * 1024);
          if (request.outputMode !== 'log') stop('Workflow command output exceeded limit');
        }
      }
      try {
        request.onOutput?.(text.slice(-16 * 1024));
      } catch {
        /* Observers do not own the process. */
      }
    };
    child.stdout.on('data', (data: Buffer) => consume(data, false));
    child.stderr.on('data', (data: Buffer) => consume(data, true));
    const cleanup = () => {
      clearTimeout(timeout);
      request.signal.removeEventListener('abort', abort);
    };
    child.once('error', (error) => {
      closed = true;
      cleanup();
      reject(error);
    });
    child.once('close', (code) => {
      closed = true;
      cleanup();
      void (async () => {
        await stopping;
        if (failure || request.signal.aborted) reject(failure ?? new Error('Workflow cancelled'));
        else if (code !== 0) reject(new Error(`Workflow command exited ${String(code)}: ${stderr.trim()}`));
        else resolve(stdout);
      })();
    });
    if (request.signal.aborted) abort();
  });
}
