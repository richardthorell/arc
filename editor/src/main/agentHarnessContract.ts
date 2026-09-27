/**
 * Transport-neutral ARC editor agent harness operation catalog.
 *
 * Built-in agents and external transports invoke the same operation names.
 * Connection adapters may expose additional connection-only operations, but
 * must not invent aliases or different editing semantics.
 */
export const agentHarnessMethods = [
  'agent.capabilities',
  'scene.overview',
  'scene.findEntities',
  'scene.getEntity',
  'scene.componentSchemas',
  'scene.spatialQuery',
  'scene.changes',
  'assets.list',
  'viewport.state',
  'viewport.move',
  'viewport.setRenderOptions',
  'viewport.pick',
  'viewport.observe',
  'viewport.debug',
  'viewport.inspectPixel',
  'viewport.compare',
  'diagnostics.get',
  'events.wait',
  'edit.request',
  'edit.begin',
  'edit.apply',
  'edit.commit',
  'edit.cancel',
  'history.undo',
  'history.redo',
] as const;

export type AgentHarnessMethod = (typeof agentHarnessMethods)[number];

export const agentEditActions = [
  'create',
  'rename',
  'setActive',
  'setTag',
  'setMobility',
  'setTransform',
  'setRenderLayer',
  'setMaterial',
  'setFlow',
  'snapToFloor',
  'delete',
  'duplicate',
  'reparent',
  'patchComponent',
  'createAsset',
] as const;

export type AgentEditAction = (typeof agentEditActions)[number];

export const gatewayMethods = ['gateway.status', ...agentHarnessMethods] as const;

export type GatewayMethod = (typeof gatewayMethods)[number];

export const gatewayHttpMethods = {
  '/api/v1/agent/capabilities': 'agent.capabilities',
  '/api/v1/scene/overview': 'scene.overview',
  '/api/v1/scene/find-entities': 'scene.findEntities',
  '/api/v1/scene/entity': 'scene.getEntity',
  '/api/v1/scene/component-schemas': 'scene.componentSchemas',
  '/api/v1/scene/spatial-query': 'scene.spatialQuery',
  '/api/v1/scene/changes': 'scene.changes',
  '/api/v1/assets/list': 'assets.list',
  '/api/v1/viewport/state': 'viewport.state',
  '/api/v1/viewport/move': 'viewport.move',
  '/api/v1/viewport/render-options': 'viewport.setRenderOptions',
  '/api/v1/viewport/pick': 'viewport.pick',
  '/api/v1/viewport/observe': 'viewport.observe',
  '/api/v1/viewport/debug': 'viewport.debug',
  '/api/v1/viewport/inspect-pixel': 'viewport.inspectPixel',
  '/api/v1/viewport/compare': 'viewport.compare',
  '/api/v1/diagnostics': 'diagnostics.get',
  '/api/v1/events/wait': 'events.wait',
  '/api/v1/edit/request': 'edit.request',
  '/api/v1/edit/begin': 'edit.begin',
  '/api/v1/edit/apply': 'edit.apply',
  '/api/v1/edit/commit': 'edit.commit',
  '/api/v1/edit/cancel': 'edit.cancel',
  '/api/v1/history/undo': 'history.undo',
  '/api/v1/history/redo': 'history.redo',
} as const satisfies Readonly<Record<string, GatewayMethod>>;
