import { useMemo, useState } from 'react';
import { Box, Database, Search } from 'lucide-react';

import type { AssetItem, SceneEntity } from '../services/editorHostTypes';

import '../tools/tools.css';

type SearchToken = { key: string; value: string };

const parseQuery = (query: string): { text: string; tokens: SearchToken[] } => {
  const tokens: SearchToken[] = [];
  const text: string[] = [];
  for (const part of query.trim().split(/\s+/).filter(Boolean)) {
    const separator = part.indexOf(':');
    if (separator > 0)
      tokens.push({ key: part.slice(0, separator).toLocaleLowerCase(), value: part.slice(separator + 1) });
    else text.push(part);
  }
  return { text: text.join(' ').toLocaleLowerCase(), tokens };
};

export function SearchPanel({
  entities,
  assets,
  onSelectEntity,
  onSelectAsset,
}: {
  entities: SceneEntity[];
  assets: AssetItem[];
  onSelectEntity: (id: string) => void;
  onSelectAsset: (id: string) => void;
}) {
  const [query, setQuery] = useState('');
  const parsed = useMemo(() => parseQuery(query), [query]);
  const flatEntities = useMemo(() => {
    const visit = (values: SceneEntity[], parentPath = ''): Array<{ entity: SceneEntity; path: string }> =>
      values.flatMap((entity) => {
        const currentPath = parentPath ? `${parentPath} / ${entity.name}` : entity.name;
        return [{ entity, path: currentPath }, ...visit(entity.children ?? [], currentPath)];
      });
    return visit(entities);
  }, [entities]);
  const entityResults = flatEntities.filter(({ entity, path }) => {
    if (parsed.text && !`${entity.name} ${path} ${entity.kind}`.toLocaleLowerCase().includes(parsed.text)) return false;
    return parsed.tokens.every((token) => {
      if (token.key === 'type') return entity.kind.toLocaleLowerCase().includes(token.value.toLocaleLowerCase());
      if (token.key === 'tag' || token.key === 'component')
        return (entity.components ?? []).some((component) =>
          component.toLocaleLowerCase().includes(token.value.toLocaleLowerCase()),
        );
      return token.key !== 'status' && token.key !== 'ref';
    });
  });
  const assetResults = assets.filter((asset) => {
    if (parsed.text && !`${asset.name} ${asset.path} ${asset.kind}`.toLocaleLowerCase().includes(parsed.text))
      return false;
    return parsed.tokens.every((token) => {
      if (token.key === 'type') return asset.kind.toLocaleLowerCase().includes(token.value.toLocaleLowerCase());
      if (token.key === 'status') return asset.status.toLocaleLowerCase().includes(token.value.toLocaleLowerCase());
      return token.key !== 'tag' && token.key !== 'component';
    });
  });

  return (
    <section className="production-tool-panel search-production-panel">
      <label className="tool-search search-hero">
        <Search size={16} />
        <input
          aria-label="Search project"
          placeholder="Search entities and assets · type:mesh status:failed component:camera"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <div className="search-result-summary">
        {entityResults.length} entities · {assetResults.length} assets
      </div>
      <section className="search-result-section">
        <h3>Entities</h3>
        {entityResults.slice(0, 100).map(({ entity, path }) => (
          <button key={entity.guid ?? entity.id} onClick={() => onSelectEntity(entity.id)} type="button">
            <Box size={14} />
            <span>
              <strong>{entity.name}</strong>
              <small>{path}</small>
            </span>
            <em>{entity.kind}</em>
          </button>
        ))}
      </section>
      <section className="search-result-section">
        <h3>Assets</h3>
        {assetResults.slice(0, 100).map((asset) => (
          <button key={asset.id} onClick={() => onSelectAsset(asset.id)} type="button">
            <Database size={14} />
            <span>
              <strong>{asset.name}</strong>
              <small>{asset.path}</small>
            </span>
            <em>{asset.status}</em>
          </button>
        ))}
      </section>
    </section>
  );
}
