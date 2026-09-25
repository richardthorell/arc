import { useEffect, useState } from 'react';

import type { WorkbenchLayoutState } from './workbenchTypes';

const layoutStorageKey = 'arc.editor.workbench.layout.v2';

type PersistedWorkbenchLayout = Omit<WorkbenchLayoutState, 'activityExpanded'>;

export const defaultWorkbenchLayout: WorkbenchLayoutState = {
  activeActivity: 'scene',
  activeCenterPanel: 'viewport',
  activeRightPanel: 'inspector',
  activeBottomPanel: 'console',
  activityExpanded: false,
  leftPanelWidth: 300,
  rightPanelWidth: 410,
  bottomPanelHeight: 148,
  leftVisible: true,
  rightVisible: true,
  bottomVisible: true,
};

const persistedLayout = (layout: WorkbenchLayoutState): PersistedWorkbenchLayout => ({
  activeActivity: layout.activeActivity,
  activeCenterPanel: layout.activeCenterPanel,
  activeRightPanel: layout.activeRightPanel,
  activeBottomPanel: layout.activeBottomPanel,
  leftPanelWidth: layout.leftPanelWidth,
  rightPanelWidth: layout.rightPanelWidth,
  bottomPanelHeight: layout.bottomPanelHeight,
  leftVisible: layout.leftVisible,
  rightVisible: layout.rightVisible,
  bottomVisible: layout.bottomVisible,
});

const readLayout = (): WorkbenchLayoutState => {
  try {
    const saved = window.localStorage.getItem(layoutStorageKey);
    if (!saved) {
      return defaultWorkbenchLayout;
    }

    return {
      ...defaultWorkbenchLayout,
      ...(JSON.parse(saved) as Partial<PersistedWorkbenchLayout>),
      activityExpanded: false,
    };
  } catch {
    return defaultWorkbenchLayout;
  }
};

export const useWorkbenchLayout = () => {
  const [layout, setLayout] = useState<WorkbenchLayoutState>(() => readLayout());

  useEffect(() => {
    window.localStorage.setItem(layoutStorageKey, JSON.stringify(persistedLayout(layout)));
  }, [layout]);

  const resetLayout = () => setLayout(defaultWorkbenchLayout);

  return {
    layout,
    setLayout,
    resetLayout,
  };
};
