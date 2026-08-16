import type { InspectorEntitySnapshot } from '../inspector/inspectorTypes';
import { panelInspectorFixture as basePanelInspectorFixture } from './UiLabPanelFixturesBase';

export * from './UiLabPanelFixturesBase';

export const panelInspectorFixture: InspectorEntitySnapshot = {
  ...basePanelInspectorFixture,
  meshRenderer: basePanelInspectorFixture.meshRenderer
    ? {
        ...basePanelInspectorFixture.meshRenderer,
        hasMesh: true,
        assetBackedMesh: true,
        meshName: 'SM_Cabin',
        meshPath: 'Assets/Environment/Cabins/SM_Cabin.glb',
      }
    : null,
};
