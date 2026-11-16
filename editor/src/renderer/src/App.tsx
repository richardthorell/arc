import { useEffect, useState } from 'react';
import type { ArcProjectCandidate } from '../../common/projectTypes';
import { Workbench } from './app/Workbench';
import { CloseDialog, type CloseDialogChoice } from './dialogs/CloseDialog';
import { ProjectBrowser } from './project/ProjectBrowser';
import { UiLabWindow } from './uiLab/UiLabWindow';
import { clearProjectSettingsCache, refreshProjectSettings } from './settings/projectSettingsCache';

const isUiLabMode = () =>
  import.meta.env.VITE_ARC_UI_LAB === '1' || new URLSearchParams(window.location.search).get('mode') === 'ui-lab';

type CloseRequest = { sceneName: string };

function EditorApplication() {
  const [project, setProject] = useState<ArcProjectCandidate | null | undefined>(undefined);
  const [closeRequest, setCloseRequest] = useState<CloseRequest | null>(null);

  useEffect(() => {
    let disposed = false;
    clearProjectSettingsCache();
    void window.arc.projects.snapshot().then(async (snapshot) => {
      if (snapshot?.activeProject) await refreshProjectSettings().catch(() => {});
      if (!disposed) setProject(snapshot?.activeProject ?? null);
    });
    const unsubscribe = window.arc.nativeWindow.onCloseRequested((request) => setCloseRequest(request));
    return () => {
      disposed = true;
      clearProjectSettingsCache();
      unsubscribe();
    };
  }, []);

  const resolveClose = (choice: CloseDialogChoice) => {
    setCloseRequest(null);
    window.arc.nativeWindow.respondToClose(choice);
  };

  let content;
  if (project === undefined) content = <div className="application-loading">Starting ARC Editor…</div>;
  else if (!project)
    content = (
      <ProjectBrowser
        onOpened={(opened) => {
          clearProjectSettingsCache();
          void refreshProjectSettings()
            .catch(() => {})
            .then(() => setProject(opened));
        }}
      />
    );
  else
    content = (
      <Workbench
        onProjectClosed={() => {
          clearProjectSettingsCache();
          setProject(null);
        }}
      />
    );

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
