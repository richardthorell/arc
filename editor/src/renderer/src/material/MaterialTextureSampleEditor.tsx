import { useCallback, useEffect, useMemo, useState } from 'react';

import { TexturePicker, type AssetPickerItem } from '../inspector/AssetPicker';
import type { MaterialGraphNode } from './materialGraphTypes';
import './materialTextureSample.css';

type HostResponse<T> = {
  succeeded: boolean;
  payload?: T;
};

type HostAssetSnapshot = {
  guid: string;
  path: string;
  scope: 'builtin' | 'project' | 'user' | 'organization';
  readOnly: boolean;
  kind: 'material' | 'texture' | 'shader' | 'mesh' | 'prefab' | 'folder' | 'environment' | 'unknown';
  typeId: string;
  importerId: string;
  state: 'unknown' | 'queued' | 'importing' | 'ready' | 'stale' | 'failed';
};

type HostProjectAssetsSnapshot = {
  assets: HostAssetSnapshot[];
};

type HostAssetThumbnailSnapshot = {
  dataUrl: string;
};

const nameFromPath = (path: string) => path.split(/[\\/]/).pop() || path;
const pickerStatus = (state: HostAssetSnapshot['state']): AssetPickerItem['status'] =>
  state === 'importing' || state === 'ready' || state === 'stale' || state === 'failed' ? state : 'unknown';

export function MaterialTextureSampleEditor({
  node,
  readOnly,
  onChange,
}: {
  node: MaterialGraphNode;
  readOnly: boolean;
  onChange: (node: MaterialGraphNode) => void;
}) {
  const [assets, setAssets] = useState<AssetPickerItem[]>([]);
  const texturePath = typeof node.values.texture === 'string' ? node.values.texture : '';

  useEffect(() => {
    let mounted = true;
    if (!window.arc?.host) return () => undefined;
    void window.arc.host
      .query('project.assets')
      .then((response) => {
        if (!mounted) return;
        const result = response as HostResponse<HostProjectAssetsSnapshot>;
        if (!result.succeeded || !result.payload?.assets) return;
        setAssets(
          result.payload.assets
            .filter((asset) => asset.kind === 'texture' || asset.kind === 'environment')
            .map((asset) => ({
              id: asset.guid || asset.path,
              guid: asset.guid || undefined,
              typeId: asset.typeId || undefined,
              name: nameFromPath(asset.path),
              path: asset.path,
              kind: asset.kind === 'environment' ? 'environment' : 'texture',
              status: pickerStatus(asset.state),
              scope: asset.scope,
              readOnly: asset.readOnly,
            })),
        );
      })
      .catch(() => {
        if (mounted) setAssets([]);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const thumbnailProvider = useCallback(async (path: string): Promise<string | null> => {
    if (!window.arc?.host) return null;
    const response = (await window.arc.host.query('asset.thumbnail', {
      path,
      maxSize: 128,
    })) as HostResponse<HostAssetThumbnailSnapshot>;
    return response.succeeded && response.payload?.dataUrl ? response.payload.dataUrl : null;
  }, []);

  const selectedAsset = useMemo(() => assets.find((asset) => asset.path === texturePath), [assets, texturePath]);

  return (
    <div className="material-node-texture-picker" title={selectedAsset?.path || texturePath || 'Choose texture asset'}>
      <TexturePicker
        assets={assets}
        value={texturePath}
        label="Texture"
        allowEmpty
        thumbnailProvider={thumbnailProvider}
        onChange={(texture) => onChange({ ...node, values: { ...node.values, texture } })}
      />
      <label className="material-node-parameter-toggle">
        <input
          checked={Boolean(node.parameter?.exposed)}
          disabled={readOnly}
          type="checkbox"
          onChange={(event) =>
            onChange({
              ...node,
              parameter: {
                exposed: event.target.checked,
                name: node.parameter?.name ?? 'Texture',
              },
            })
          }
        />
        Parameter
        {node.parameter?.exposed && (
          <input
            aria-label="Parameter name"
            disabled={readOnly}
            value={node.parameter.name}
            onChange={(event) => onChange({ ...node, parameter: { ...node.parameter!, name: event.target.value } })}
          />
        )}
      </label>
      {readOnly && <span className="material-node-texture-readonly" aria-hidden="true" />}
    </div>
  );
}
