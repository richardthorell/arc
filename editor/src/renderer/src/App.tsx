import { useEffect, useState } from 'react';
import type { ArcProjectCandidate } from '../../common/projectTypes';
import { Workbench } from './app/Workbench';
import { ProjectBrowser } from './project/ProjectBrowser';
import { UiLab } from './uiLab/UiLab';

const isUiLabMode = () => new URLSearchParams(window.location.search).get('mode') === 'ui-lab';

function EditorApplication() {
  const [project, setProject] = useState<ArcProjectCandidate | null | undefined>(undefined);

  useEffect(() => {
    void window.arc.projects.snapshot().then((snapshot) => setProject(snapshot?.activeProject ?? null));
  }, []);

  if (project === undefined) return <div className="application-loading">Starting ARC Editor…</div>;
  if (!project) return <ProjectBrowser onOpened={setProject} />;
  return <Workbench onProjectClosed={() => setProject(null)} />;
}

export function App() {
  return isUiLabMode() ? <UiLab /> : <EditorApplication />;
}
