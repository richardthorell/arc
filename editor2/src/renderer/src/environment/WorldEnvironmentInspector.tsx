import { Search, SlidersHorizontal } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import type { AssetPickerItem, AssetThumbnailProvider } from '../inspector/AssetPicker';
import { SchemaComponentCard } from '../inspector/SchemaComponents';
import { setPathValue } from '../inspector/propertySchema';
import type { HostWorldEnvironment } from './environmentTypes';
import { worldEnvironmentSchemas } from './worldEnvironmentSchemas';

export type WorldEnvironmentInspectorProps = {
  environment: HostWorldEnvironment;
  assets: ReadonlyArray<AssetPickerItem>;
  thumbnailProvider?: AssetThumbnailProvider;
  onChange: (environment: HostWorldEnvironment) => void;
  onPreset: (preset: string) => void;
  onHdri: (path: string) => Promise<boolean> | boolean | void;
};

const environmentPresets = [
  ['clearDay', 'Clear Day'], ['alpineLateMorning', 'Alpine'], ['goldenHour', 'Golden Hour'],
  ['overcast', 'Overcast'], ['night', 'Night'], ['indoorNeutral', 'Indoor'],
] as const;

export function WorldEnvironmentInspector({
  environment, assets, thumbnailProvider, onChange, onPreset, onHdri,
}: WorldEnvironmentInspectorProps) {
  const [draft, setDraft] = useState(environment);
  const [filter, setFilter] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>(() => Object.fromEntries(
    worldEnvironmentSchemas.map((schema) => [schema.id, schema.collapsedByDefault ?? false]),
  ));
  useEffect(() => setDraft(environment), [environment]);

  const schemas = useMemo(() => {
    const needle = filter.trim().toLocaleLowerCase();
    return worldEnvironmentSchemas.filter((schema) => (!schema.visible || schema.visible(draft)) && (!needle ||
      schema.title.toLocaleLowerCase().includes(needle) ||
      schema.fields.some((field) => field.label.toLocaleLowerCase().includes(needle))));
  }, [draft, filter]);

  const update = (path: string, value: unknown) => {
    const next = setPathValue(draft, path, value);
    setDraft(next);
    if (path === 'hdriPath') {
      void Promise.resolve(onHdri(value as string)).then((accepted) => {
        if (accepted === false) setDraft(environment);
      });
      return;
    }
    onChange(next);
  };

  return (
    <section className="environment-inspector data-inspector">
      <div className="environment-preset-strip" aria-label="World environment presets">
        {environmentPresets.map(([id, label]) => <button key={id} onClick={() => onPreset(id)} type="button">{label}</button>)}
      </div>
      <div className="inspector-search-row environment-search-row">
        <label><Search size={15} /><input aria-label="Search world settings" onChange={(event) => setFilter(event.target.value)}
          placeholder="Search world settings…" value={filter} /></label>
        <button aria-label="World settings filter options" title="Filter virtual components" type="button"><SlidersHorizontal size={15} /></button>
      </div>
      <div className="inspector-component-list environment-component-list">
        {schemas.map((schema) => <SchemaComponentCard
          key={schema.id}
          assets={assets}
          collapsed={collapsed[schema.id] ?? false}
          context={draft}
          schema={schema}
          thumbnailProvider={thumbnailProvider}
          onToggle={() => setCollapsed((current) => ({ ...current, [schema.id]: !(current[schema.id] ?? false) }))}
          onValue={(path, value) => update(path, value)}
        />)}
        {!schemas.length && <div className="inspector-state compact">No world settings match “{filter}”.</div>}
      </div>
    </section>
  );
}
