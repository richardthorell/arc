import { useEffect, useState } from 'react';
import type { ArcProjectCandidate } from '../../common/projectTypes';
import { Workbench } from './app/Workbench';
import { CloseDialog, type CloseDialogChoice } from './dialogs/CloseDialog';
import { ProjectBrowser } from './project/ProjectBrowser';
import { UiLabWindow } from './uiLab/UiLabWindow';

const isUiLabMode = () =>
  import.meta.env.VITE_ARC_UI_LAB === '1' || new URLSearchParams(window.location.search).get('mode') === 'ui-lab';

type CloseRequest = { sceneName: string };

function EditorApplication() {
  const [project, setProject] = useState<ArcProjectCandidate | null | undefined>(undefined);
  const [closeRequest, setCloseRequest] = useState<CloseRequest | null>(null);

  useEffect(() => {
    void window.arc.projects.snapshot().then((snapshot) => setProject(snapshot?.activeProject ?? null));
    return window.arc.nativeWindow.onCloseRequested((request) => setCloseRequest(request));
  }, []);

  const resolveClose = (choice: CloseDialogChoice) => {
    setCloseRequest(null);
    window.arc.nativeWindow.respondToClose(choice);
  };

  let content;
  if (project === undefined) content = <div className="application-loading">Starting ARC Editor…</div>;
  else if (!project) content = <ProjectBrowser onOpened={setProject} />;
  else content = <Workbench onProjectClosed={() => setProject(null)} />;

  return (
    <>
      {content}
      {closeRequest && <CloseDialog onChoose={resolveClose} sceneName={closeRequest.sceneName} />}
    </>
  );
}

function UiLabApplication() {
  useEffect(
    () =>
      window.arc.nativeWindow.onCloseRequested(() => {
        window.arc.nativeWindow.respondToClose('discard');
      }),
    [],
  );

  return <UiLabWindow />;
}

export function App() {
  return isUiLabMode() ? <UiLabApplication /> : <EditorApplication />;
}
