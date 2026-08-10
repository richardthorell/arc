import { useEffect, useState } from 'react';
import { AlertTriangle, Box, FolderOpen, GitBranch, Plus, RefreshCw, Search, Settings, Trash2, X } from 'lucide-react';

import type {
  ArcProjectBrowserSnapshot,
  ArcProjectCandidate,
  ArcProjectOperationResult,
} from '../../../common/projectTypes';
import type { RecoverySnapshot } from '../../../common/editorWorkflowTypes';
import { WindowControls } from '../layout/WindowControls';
import { UiButton, UiIconButton } from '../ui';

import './projectBrowser.css';

type ProjectBrowserProps = {
  onOpened: (project: ArcProjectCandidate) => void;
};

const compareEngineVersions = (left: string, right: string): number => {
  const numeric = (value: string) =>
    value
      .replace(/^v/i, '')
      .split(/[.+-]/)
      .slice(0, 3)
      .map((part) => Number.parseInt(part, 10) || 0);
  const first = numeric(left);
  const second = numeric(right);
  for (let index = 0; index < 3; ++index) {
    if (first[index] !== second[index]) return first[index] < second[index] ? -1 : 1;
  }
  return 0;
};

export function ProjectBrowser({ onOpened }: ProjectBrowserProps) {
  const [snapshot, setSnapshot] = useState<ArcProjectBrowserSnapshot | null>(null);
  const [mode, setMode] = useState<'recent' | 'create' | 'clone'>('recent');
  const [projectName, setProjectName] = useState('New ARC Project');
  const [projectTemplate, setProjectTemplate] = useState('blank-3d');
  const [cloneSource, setCloneSource] = useState('');
  const [destination, setDestination] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [search, setSearch] = useState('');
  const [recoveries, setRecoveries] = useState<Record<string, RecoverySnapshot>>({});

  const refresh = async () => {
    const next = await window.arc.projects.snapshot();
    setSnapshot(next);
    if (!next) return;
    const entries = await Promise.all(
      next.recentProjects.map(
        async (project) =>
          [project.guid, await window.arc.recovery.snapshot(project.guid, project.projectRoot)] as const,
      ),
    );
    setRecoveries(
      Object.fromEntries(entries.filter((entry): entry is readonly [string, RecoverySnapshot] => Boolean(entry[1]))),
    );
  };

  useEffect(() => {
    void refresh();
  }, []);

  const finish = async (result: ArcProjectOperationResult) => {
    setBusy(false);
    if (result.succeeded && result.project) {
      onOpened(result.project);
      return;
    }
    setMessage(result.error || 'Project operation failed');
    await refresh();
  };

  const chooseDestination = async () => {
    const selected = await window.arc.dialog.projectDestination();
    if (selected) setDestination(selected);
  };

  const openDescriptor = async () => {
    const selected = await window.arc.dialog.openProject();
    if (!selected) return;
    setBusy(true);
    let result = await window.arc.projects.open(selected);
    const requiredVersion = result.project?.descriptor.engineVersion;
    const matchingInstallation = snapshot?.installations.find(
      (installation) => installation.version === requiredVersion && !installation.current && installation.editorPath,
    );
    if (
      !result.succeeded &&
      matchingInstallation &&
      window.confirm(`Open this project in its matching ARC ${requiredVersion} editor?`)
    ) {
      const launched = await window.arc.projects.launchMatchingEngine(selected);
      setBusy(false);
      setMessage(
        launched.succeeded
          ? `Launched ARC ${requiredVersion}.`
          : launched.error || 'Could not launch the matching editor',
      );
      return;
    }
    if (!result.succeeded && result.project?.compatibility === 'upgradeRequired') {
      const approved = window.confirm(
        `Upgrade ${result.project.descriptor.name} from ARC ${result.project.descriptor.engineVersion} to ${snapshot?.currentEngineVersion}? A validated descriptor backup will be created first.`,
      );
      if (approved) result = await window.arc.projects.open(selected, { upgrade: true });
    } else if (!result.succeeded && result.project?.compatibility === 'newerEngineRequired') {
      const approved = window.confirm(
        `${result.project.descriptor.name} targets newer ARC ${result.project.descriptor.engineVersion}. Open it read-only?`,
      );
      if (approved) result = await window.arc.projects.open(selected, { readOnly: true });
    }
    await finish(result);
  };

  const openRecent = async (descriptorPath: string, compatibility: string) => {
    const recent = snapshot?.recentProjects.find((entry) => entry.descriptorPath === descriptorPath);
    const matchingInstallation = snapshot?.installations.find(
      (installation) =>
        installation.version === recent?.engineVersion && !installation.current && installation.editorPath,
    );
    if (
      compatibility !== 'compatible' &&
      matchingInstallation &&
      window.confirm(`Open this project in ARC ${recent?.engineVersion}?`)
    ) {
      setBusy(true);
      const launched = await window.arc.projects.launchMatchingEngine(descriptorPath);
      setBusy(false);
      setMessage(
        launched.succeeded
          ? `Launched ARC ${recent?.engineVersion}.`
          : launched.error || 'Could not launch the matching editor',
      );
      return;
    }
    if (
      compatibility === 'upgradeRequired' &&
      !window.confirm(
        'Upgrade this project to the running ARC version? A validated descriptor backup is created first.',
      )
    )
      return;
    if (
      compatibility === 'newerEngineRequired' &&
      !window.confirm('This project targets a newer ARC version. Open it read-only?')
    )
      return;
    setBusy(true);
    await finish(
      await window.arc.projects.open(descriptorPath, {
        upgrade: compatibility === 'upgradeRequired',
        readOnly: compatibility === 'newerEngineRequired',
      }),
    );
  };

  const recoverRecent = async (descriptorPath: string, compatibility: string, recovery: RecoverySnapshot) => {
    const latest = recovery.generations[0];
    if (!latest) return;
    setBusy(true);
    const opened = await window.arc.projects.open(descriptorPath, {
      upgrade: compatibility === 'upgradeRequired',
      readOnly: compatibility === 'newerEngineRequired',
    });
    if (!opened.succeeded || !opened.project) return finish(opened);
    const restored = (await window.arc.recovery.restore(latest.id)) as { succeeded?: boolean; error?: string };
    if (!restored?.succeeded) {
      setBusy(false);
      return setMessage(restored?.error || 'Recovery could not be opened');
    }
    setBusy(false);
    onOpened(opened.project);
  };

  const createProject = async () => {
    if (!projectName.trim() || !destination.trim()) return setMessage('Project name and destination are required');
    setBusy(true);
    const created = await window.arc.projects.create({
      name: projectName.trim(),
      destination,
      template: projectTemplate,
    });
    if (!created.succeeded || !created.project) return finish(created);
    await finish(await window.arc.projects.open(created.project.descriptorPath));
  };

  const cloneProject = async () => {
    if (!cloneSource.trim() || !destination.trim()) return setMessage('Clone source and destination are required');
    setBusy(true);
    const cloned = await window.arc.projects.clone({ source: cloneSource.trim(), destination });
    if (!cloned.succeeded || !cloned.project) return finish(cloned);
    if (
      cloned.project.compatibility === 'upgradeRequired' &&
      !window.confirm('Upgrade the cloned project to the running ARC version? A descriptor backup is created first.')
    )
      return finish({ succeeded: false, error: 'Project upgrade cancelled', project: cloned.project });
    if (
      cloned.project.compatibility === 'newerEngineRequired' &&
      !window.confirm('The cloned project targets a newer ARC version. Open it read-only?')
    )
      return finish({ succeeded: false, error: 'Read-only open cancelled', project: cloned.project });
    await finish(
      await window.arc.projects.open(cloned.project.descriptorPath, {
        upgrade: cloned.project.compatibility === 'upgradeRequired',
        readOnly: cloned.project.compatibility === 'newerEngineRequired',
      }),
    );
  };

  const removeRecent = async (descriptorPath: string) => {
    await window.arc.projects.removeRecent(descriptorPath);
    await refresh();
  };

  const deleteProject = async (recent: ArcProjectBrowserSnapshot['recentProjects'][number]) => {
    if (
      !window.confirm(
        `Move “${recent.name}” and every file under\n${recent.projectRoot}\n\nto the system Trash or Recycle Bin?`,
      )
    )
      return;
    setBusy(true);
    const result = await window.arc.projects.delete(recent.descriptorPath);
    setBusy(false);
    setMessage(result.succeeded ? `Moved ${recent.name} to the trash.` : result.error || 'Could not delete project');
    await refresh();
  };

  const recentProjects = (snapshot?.recentProjects ?? []).filter((project) => {
    const query = search.trim().toLocaleLowerCase();
    return (
      !query ||
      project.name.toLocaleLowerCase().includes(query) ||
      project.projectRoot.toLocaleLowerCase().includes(query)
    );
  });

  return (
    <main className="project-browser-shell">
      <header className="project-browser-header">
        <div className="project-browser-brand">
          <span className="project-browser-logo">
            <Box size={23} />
          </span>
          <div>
            <strong>ARC</strong>
            <span>Production Editor</span>
          </div>
        </div>
        <span className="project-browser-engine-version">Engine {snapshot?.currentEngineVersion ?? '...'}</span>
        <WindowControls />
      </header>

      <section className="project-browser-body">
        <aside className="project-browser-nav">
          <button className={mode === 'recent' ? 'active' : ''} onClick={() => setMode('recent')} type="button">
            <FolderOpen size={17} /> Projects
          </button>
          <button className={mode === 'create' ? 'active' : ''} onClick={() => setMode('create')} type="button">
            <Plus size={17} /> Create
          </button>
          <button className={mode === 'clone' ? 'active' : ''} onClick={() => setMode('clone')} type="button">
            <GitBranch size={17} /> Clone
          </button>
          <button
            className="project-browser-settings"
            onClick={() => setMessage('Editor settings are available after opening a project.')}
            type="button"
          >
            <Settings size={17} /> Settings
          </button>
        </aside>

        <section className="project-browser-content">
          <div className="project-browser-title">
            <div>
              <h1>{mode === 'recent' ? 'Welcome to ARC' : mode === 'create' ? 'Create project' : 'Clone project'}</h1>
              <p>
                {mode === 'recent'
                  ? 'Open a recent workspace or create a new ARC project.'
                  : mode === 'create'
                    ? 'Create a complete external C++ repository from an installed ARC template.'
                    : 'Clone a Git repository or copy a local project directory.'}
              </p>
            </div>
            {mode === 'recent' && (
              <div className="project-browser-actions">
                <button
                  className="project-browser-open-action"
                  disabled={busy}
                  onClick={() => void openDescriptor()}
                  type="button"
                >
                  <FolderOpen size={18} /> Open Project
                </button>
                <button className="project-browser-create-action" onClick={() => setMode('create')} type="button">
                  <Plus size={20} /> Create Project
                </button>
              </div>
            )}
          </div>

          {!snapshot?.hostConnected && (
            <div className="project-browser-error">
              <AlertTriangle size={16} />
              <div>
                <strong>Native editor host unavailable</strong>
                <span>{snapshot?.hostError || 'Build arc_host_process before opening a project.'}</span>
              </div>
            </div>
          )}

          {mode === 'recent' && (
            <div className="project-list">
              <div className="project-list-header">
                <h2>Recent Projects</h2>
                <div className="project-list-tools">
                  <button aria-label="Refresh projects" onClick={() => void refresh()} type="button">
                    <RefreshCw size={17} />
                  </button>
                  <label className="project-browser-search">
                    <Search size={16} />
                    <input
                      aria-label="Search projects"
                      onChange={(event) => setSearch(event.target.value)}
                      placeholder="Search projects..."
                      value={search}
                    />
                  </label>
                </div>
              </div>
              {recentProjects.map((recent) => {
                const recovery = recoveries[recent.guid];
                const comparison = compareEngineVersions(
                  recent.engineVersion,
                  snapshot?.currentEngineVersion ?? recent.engineVersion,
                );
                const compatibility =
                  comparison === 0 ? 'compatible' : comparison < 0 ? 'upgradeRequired' : 'newerEngineRequired';
                return (
                  <article className="project-card" data-missing={recent.missing} key={recent.guid}>
                    <button
                      disabled={busy || recent.missing || !snapshot?.hostConnected}
                      onClick={() => {
                        void openRecent(recent.descriptorPath, compatibility);
                      }}
                      type="button"
                    >
                      <span className="project-card-icon">
                        <Box size={30} strokeWidth={1.7} />
                      </span>
                      <span className="project-card-copy">
                        <strong>{recent.name}</strong>
                        <span>{recent.projectRoot}</span>
                        <small>
                          ARC {recent.engineVersion} ·{' '}
                          {recent.missing ? 'Missing' : new Date(recent.lastOpenedAt).toLocaleString()}
                        </small>
                      </span>
                    </button>
                    <div className="project-card-controls">
                      {recovery?.uncleanShutdown && recovery.generations.length > 0 && (
                        <UiButton
                          className="project-recovery-button"
                          disabled={busy || recent.missing || !snapshot?.hostConnected}
                          onClick={() => void recoverRecent(recent.descriptorPath, compatibility, recovery)}
                          variant="toolbar"
                        >
                          Recover
                        </UiButton>
                      )}
                      <UiIconButton
                        className="project-card-remove"
                        label={`Remove ${recent.name} from Recent Projects`}
                        onClick={() => void removeRecent(recent.descriptorPath)}
                      >
                        <X size={18} />
                      </UiIconButton>
                      {!recent.missing && (
                        <UiButton
                          className="project-card-delete"
                          disabled={busy}
                          onClick={() => void deleteProject(recent)}
                          title={`Move ${recent.name} to Trash`}
                          variant="toolbar"
                        >
                          <Trash2 size={16} /> Delete
                        </UiButton>
                      )}
                    </div>
                  </article>
                );
              })}
              {!recentProjects.length && (
                <div className="project-browser-empty">
                  <FolderOpen size={40} strokeWidth={1.3} />
                  <strong>{search ? 'No matching projects' : 'No projects yet'}</strong>
                  <span>
                    {search
                      ? 'Try a different project name or location.'
                      : 'Create a new project or open an existing one to get started.'}
                  </span>
                </div>
              )}
            </div>
          )}

          {mode !== 'recent' && (
            <div className="project-form">
              {mode === 'create' ? (
                <>
                  <label>
                    Project name
                    <input value={projectName} onChange={(event) => setProjectName(event.target.value)} />
                  </label>
                  <label>
                    Project template
                    <select value={projectTemplate} onChange={(event) => setProjectTemplate(event.target.value)}>
                      {(snapshot?.templates ?? []).map((template) => (
                        <option key={template.id} value={template.id}>
                          {template.name}
                        </option>
                      ))}
                    </select>
                    <small>
                      {snapshot?.templates.find((template) => template.id === projectTemplate)?.description}
                    </small>
                  </label>
                </>
              ) : (
                <label>
                  Git URL or local project directory
                  <input
                    placeholder="https://example.com/team/project.git"
                    value={cloneSource}
                    onChange={(event) => setCloneSource(event.target.value)}
                  />
                </label>
              )}
              <label>
                Empty destination directory
                <span className="project-path-input">
                  <input value={destination} onChange={(event) => setDestination(event.target.value)} />
                  <UiButton onClick={() => void chooseDestination()} variant="toolbar">
                    Browse
                  </UiButton>
                </span>
              </label>
              <div className="project-form-actions">
                <UiButton
                  disabled={busy || !snapshot?.hostConnected}
                  onClick={() => void (mode === 'create' ? createProject() : cloneProject())}
                  variant="primary"
                >
                  {busy ? 'Working…' : mode === 'create' ? 'Create and open' : 'Clone and open'}
                </UiButton>
              </div>
            </div>
          )}

          {message && <div className="project-browser-message">{message}</div>}
        </section>
      </section>
    </main>
  );
}
