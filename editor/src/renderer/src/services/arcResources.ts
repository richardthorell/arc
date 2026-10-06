import { createArcAssetResourceHandler, createWindowArcAssetResourceEnvironment } from './arcAssetResources';
import { ArcResourceRegistry } from './arcResourceRegistry';

export const createWindowArcResourceRegistry = (): ArcResourceRegistry => {
  const registry = new ArcResourceRegistry();
  registry.register(createArcAssetResourceHandler(createWindowArcAssetResourceEnvironment()));
  return registry;
};
