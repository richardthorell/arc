import { describe, expect, it, vi } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import {
  applyAssetMetadata,
  assetMetadataFilePath,
  loadAssetMetadata,
  normalizeAssetTags,
  parseAssetMetadata,
  saveAssetMetadata,
  serializeAssetMetadata,
  setAssetMetadataEntry,
} from './assetMetadataStore';

const asset: AssetItem = {
  id: 'rock',
  guid: 'ROCK-GUID',
  name: 'rock.glb',
  path: 'Content/Props/rock.glb',
  kind: 'mesh',
  status: 'ready',
};

describe('assetMetadataStore', () => {
  it('normalizes tags without losing the first user-facing spelling', () => {
    expect(normalizeAssetTags([' Hero ', 'hero', '', 'Environment', ' environment '])).toEqual(['Hero', 'Environment']);
  });

  it('serializes project metadata deterministically and ignores empty entries', () => {
    const entries = setAssetMetadataEntry({}, asset, {
      title: ' Hero Rock ',
      description: ' Main landmark ',
      tags: ['Hero', ' hero ', 'Props'],
    });
    const text = serializeAssetMetadata({ empty: {}, ...entries });

    expect(parseAssetMetadata(text)).toEqual({
      'guid:rock-guid': {
        title: 'Hero Rock',
        description: 'Main landmark',
        tags: ['Hero', 'Props'],
      },
    });
    expect(text).not.toContain('"empty"');
  });

  it('overlays metadata without changing storage identity', () => {
    const [updated] = applyAssetMetadata([asset], {
      'guid:rock-guid': { title: 'Cliff Rock', tags: ['landmark'] },
    });

    expect(updated.title).toBe('Cliff Rock');
    expect(updated.tags).toEqual(['landmark']);
    expect(updated.guid).toBe(asset.guid);
    expect(updated.path).toBe(asset.path);
  });

  it('loads a missing metadata file as an empty project metadata set', async () => {
    const api = {
      readText: vi.fn().mockRejectedValue(new Error('missing')),
      writeText: vi.fn(),
    };

    await expect(loadAssetMetadata(api)).resolves.toEqual({});
    expect(api.readText).toHaveBeenCalledWith(assetMetadataFilePath);
  });

  it('persists metadata in the project file instead of view state', async () => {
    const api = {
      readText: vi.fn(),
      writeText: vi.fn().mockResolvedValue({ succeeded: true }),
    };
    const entries = setAssetMetadataEntry({}, asset, { title: 'Hero Rock', tags: ['hero'] });

    await saveAssetMetadata(entries, api);

    expect(api.writeText).toHaveBeenCalledOnce();
    expect(api.writeText.mock.calls[0][0]).toBe(assetMetadataFilePath);
    expect(api.writeText.mock.calls[0][1]).toContain('Hero Rock');
  });
});
