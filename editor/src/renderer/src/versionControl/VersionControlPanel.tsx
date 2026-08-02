import { useEffect, useState } from 'react';
import { Check, GitBranch, RefreshCw, RotateCcw } from 'lucide-react';

import type { SourceControlFile, SourceControlSnapshot } from '../../../common/editorWorkflowTypes';
import { UiButton, UiIconButton } from '../ui';

import '../tools/tools.css';

export function VersionControlPanel() {
  const [snapshot, setSnapshot] = useState<SourceControlSnapshot | null>(null);
  const [selected, setSelected] = useState<SourceControlFile | null>(null);
  const [diff, setDiff] = useState('');
  const [commitMessage, setCommitMessage] = useState('');
  const [message, setMessage] = useState('');

  const refresh = async () => {
    const next = await window.arc.sourceControl.snapshot();
    if (next) setSnapshot(next);
  };
  useEffect(() => {
    void refresh();
  }, []);

  const run = async (operation: Promise<{ succeeded: boolean; output: string; error: string } | undefined>) => {
    const result = await operation;
    setMessage(
      result?.succeeded ? result.output.trim() || 'Source-control operation completed' : result?.error || 'Failed',
    );
    await refresh();
  };

  const select = async (file: SourceControlFile) => {
    setSelected(file);
    const result = await window.arc.sourceControl.diff(file.path, Boolean(file.indexState && !file.worktreeState));
    setDiff(result?.succeeded ? result.output : result?.error || '');
  };

  return (
    <section className="production-tool-panel vcs-panel">
      <header className="tool-panel-toolbar">
        <GitBranch size={15} />
        <strong>{snapshot?.branch || 'Version Control'}</strong>
        {snapshot?.available && (
          <span>
            ↑{snapshot.ahead} ↓{snapshot.behind}
          </span>
        )}
        <UiIconButton label="Refresh source control" onClick={() => void refresh()}>
          <RefreshCw size={14} />
        </UiIconButton>
        <UiButton onClick={() => void run(window.arc.sourceControl.pull())} variant="toolbar">
          Pull
        </UiButton>
        <UiButton onClick={() => void run(window.arc.sourceControl.push())} variant="toolbar">
          Push
        </UiButton>
      </header>
      {!snapshot?.available ? (
        <div className="tool-empty">{snapshot?.error || 'Checking repository…'}</div>
      ) : (
        <div className="vcs-layout">
          <div className="vcs-file-list">
            {snapshot.files.map((file) => (
              <button
                className={selected?.path === file.path ? 'selected' : ''}
                key={file.path}
                onClick={() => void select(file)}
                type="button"
              >
                <span>{file.indexState || file.worktreeState}</span>
                <strong>{file.path}</strong>
              </button>
            ))}
            {!snapshot.files.length && <div className="tool-empty">Working tree is clean.</div>}
          </div>
          <div className="vcs-diff">
            <header>
              <strong>{selected?.path || 'Select a changed file'}</strong>
              {selected && (
                <>
                  <UiButton
                    onClick={() =>
                      void run(
                        selected.indexState
                          ? window.arc.sourceControl.unstage([selected.path])
                          : window.arc.sourceControl.stage([selected.path]),
                      )
                    }
                    variant="toolbar"
                  >
                    <Check size={13} /> {selected.indexState ? 'Unstage' : 'Stage'}
                  </UiButton>
                  <UiButton
                    onClick={() => {
                      if (window.confirm(`Discard working-tree changes to ${selected.path}? This cannot be undone.`))
                        void run(window.arc.sourceControl.discard([selected.path]));
                    }}
                    variant="toolbar"
                  >
                    <RotateCcw size={13} /> Discard
                  </UiButton>
                </>
              )}
            </header>
            <pre>{diff || 'No textual diff is available.'}</pre>
          </div>
          <div className="vcs-commit">
            <textarea
              aria-label="Commit message"
              placeholder="Commit message"
              value={commitMessage}
              onChange={(event) => setCommitMessage(event.target.value)}
            />
            <UiButton
              disabled={!commitMessage.trim()}
              onClick={() => void run(window.arc.sourceControl.commit(commitMessage)).then(() => setCommitMessage(''))}
              variant="primary"
            >
              Commit staged changes
            </UiButton>
          </div>
        </div>
      )}
      {message && <div className="tool-message">{message}</div>}
    </section>
  );
}
