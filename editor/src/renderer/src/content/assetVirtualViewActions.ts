import { AssetVirtualView, AssetVirtualViewKind } from './assetVirtualViews';

export type AssetVirtualViewAction = 'open' | 'reveal' | 'rename' | 'move' | 'delete' | 'remove-from-view';

export interface AssetVirtualViewActionAsset {
  assetId: string;
  writable?: boolean;
}

export interface AssetVirtualViewActionState {
  action: AssetVirtualViewAction;
  enabled: boolean;
}

const mutableAssetActions = new Set<AssetVirtualViewAction>(['rename', 'move', 'delete']);

/**
 * Projects the normal asset action set into a virtual view without turning the
 * view into a storage scope. Asset mutability still comes from the resolved
 * asset itself; virtual membership only adds the collection-specific remove
 * action for durable collections.
 */
export function getAssetVirtualViewActions(
  view: AssetVirtualView,
  asset: AssetVirtualViewActionAsset,
): AssetVirtualViewActionState[] {
  const actions: AssetVirtualViewAction[] = ['open', 'reveal', 'rename', 'move', 'delete'];
  if (supportsMembershipRemoval(view.kind)) {
    actions.push('remove-from-view');
  }

  return actions.map((action) => ({
    action,
    enabled: !mutableAssetActions.has(action) || asset.writable !== false,
  }));
}

function supportsMembershipRemoval(kind: AssetVirtualViewKind): boolean {
  return kind === 'favorites' || kind === 'downloads';
}
