import type { AssetItem } from '../services/editorHostTypes';

const normalizeSearchText = (value: string) => value.trim().toLocaleLowerCase();

export const assetMatchesMetadataSearch = (asset: AssetItem, query: string): boolean => {
  const terms = normalizeSearchText(query).split(/\s+/).filter(Boolean);
  if (terms.length === 0) return true;

  const searchable = normalizeSearchText(
    [asset.name, asset.title, asset.description, asset.path, asset.guid, ...(asset.tags ?? [])]
      .filter((value): value is string => Boolean(value))
      .join(' '),
  );

  return terms.every((term) => searchable.includes(term));
};
