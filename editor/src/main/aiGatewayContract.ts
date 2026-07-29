/**
 * Transport-neutral ARC AI Scene Gateway operation catalog.
 *
 * MCP tools, JSON-RPC, OpenAPI metadata, and direct HTTP routes all resolve to
 * these names. Keep operation behavior in SceneGatewayCore; adapters must not
 * invent aliases or validation semantics.
 */
export const gatewayMethods = [
  'gateway.status',
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

export type GatewayMethod = (typeof gatewayMethods)[number];

export const gatewayHttpMethods = {
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
