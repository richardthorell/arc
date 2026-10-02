import { describe, expect, it } from 'vitest'

import {
  addAssetToVirtualView,
  createAssetVirtualView,
  removeAssetFromVirtualView,
  resolveAssetVirtualView,
} from './assetVirtualViews'

describe('assetVirtualViews', () => {
  it('keeps virtual membership separate from asset identity', () => {
    const favorite = createAssetVirtualView('favorites', ['asset-a', 'asset-b'])
    const recent = createAssetVirtualView('recent', ['asset-b', 'asset-a'])
    const assets = [
      { assetId: 'asset-a', path: 'Project/A.arcasset' },
      { assetId: 'asset-b', path: 'User/B.arcasset' },
    ]

    expect(resolveAssetVirtualView(favorite, assets)).toEqual([assets[0], assets[1]])
    expect(resolveAssetVirtualView(recent, assets)).toEqual([assets[1], assets[0]])
    expect(assets.map((asset) => asset.assetId)).toEqual(['asset-a', 'asset-b'])
  })

  it('normalizes duplicate and empty membership without changing order', () => {
    const view = createAssetVirtualView('downloads', [' asset-a ', '', 'asset-b', 'asset-a'])
    expect(view.assetIds).toEqual(['asset-a', 'asset-b'])
    expect(view.transient).toBe(false)
  })

  it('marks search results transient while persistent collections remain durable', () => {
    expect(createAssetVirtualView('search-results', ['asset-a']).transient).toBe(true)
    expect(createAssetVirtualView('favorites', ['asset-a']).transient).toBe(false)
  })

  it('preserves normal asset objects and silently omits stale membership', () => {
    const asset = { assetId: 'asset-a', writable: false }
    const view = createAssetVirtualView('favorites', ['missing', 'asset-a'])
    expect(resolveAssetVirtualView(view, [asset])).toEqual([asset])
  })

  it('adds and removes membership immutably', () => {
    const original = createAssetVirtualView('favorites', ['asset-a'])
    const added = addAssetToVirtualView(original, 'asset-b')
    const removed = removeAssetFromVirtualView(added, 'asset-a')

    expect(original.assetIds).toEqual(['asset-a'])
    expect(added.assetIds).toEqual(['asset-a', 'asset-b'])
    expect(removed.assetIds).toEqual(['asset-b'])
    expect(addAssetToVirtualView(added, 'asset-b')).toBe(added)
    expect(removeAssetFromVirtualView(removed, 'missing')).toBe(removed)
  })
})
