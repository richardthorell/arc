import { useEffect, useState } from 'react';

import type { WorkbenchLayoutState } from './workbenchTypes';

const layoutStorageKey = 'arc.editor2.workbench.layout.v1';

export const defaultWorkbenchLayout: WorkbenchLayoutState = {
  activeActivity: 'scene',
  activeCenterPanel: 'viewport',
  activeRightPanel: 'inspector',
  activeBottomPanel: 'console',
  activityExpanded: false,
  leftPanelWidth: 300,
  rightPanelWidth: 320,
  bottomPanelHeight: 148,
  leftVisible: true,
  rightVisible: true,
  bottomVisible: true,
};

const readLayout = (): WorkbenchLayoutState => {
  try {
    const saved = window.localStorage.getItem(layoutStorageKey);
    if (!saved) {
      return defaultWorkbenchLayout;
    }

    return {
      ...defaultWorkbenchLayout,
      ...JSON.parse(saved),
    };
  } catch {
    return defaultWorkbenchLayout;
  }
};

export const useWorkbenchLayout = () => {
  const [layout, setLayout] = useState<WorkbenchLayoutState>(() => readLayout());

  useEffect(() => {
    window.localStorage.setItem(layoutStorageKey, JSON.stringify(layout));
  }, [layout]);

  const resetLayout = () => setLayout(defaultWorkbenchLayout);

  return {
    layout,
    setLayout,
    resetLayout,
  };
};
