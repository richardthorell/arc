import { Plus, Search } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import type { HostProjectComponentSchema } from './componentSchemas';
import type { InspectorEntitySnapshot } from './inspectorTypes';

import './addComponentPicker.css';

type AddComponentOption = {
  id: string;
  label: string;
  category: string;
  tooltip?: string;
};

type AddComponentPickerProps = {
  snapshot: InspectorEntitySnapshot;
  projectSchemas: ReadonlyArray<HostProjectComponentSchema>;
  onAdd: (component: string, label: string) => Promise<boolean>;
};

const builtInOptions: ReadonlyArray<AddComponentOption> = [
  { id: 'camera', label: 'Camera', category: 'Rendering' },
  { id: 'meshRenderer', label: 'Mesh Renderer', category: 'Rendering' },
  { id: 'directionalLight', label: 'Directional Light', category: 'Lighting' },
  { id: 'pointLight', label: 'Point Light', category: 'Lighting' },
  { id: 'spotLight', label: 'Spot Light', category: 'Lighting' },
  { id: 'areaLight', label: 'Area Light', category: 'Lighting' },
  { id: 'terrain', label: 'Terrain', category: 'World' },
];

function builtInAlreadyPresent(snapshot: InspectorEntitySnapshot, id: string) {
  if (id === 'camera') return snapshot.camera !== null;
  if (id === 'meshRenderer') return snapshot.meshRenderer !== null;
  if (id.endsWith('Light')) return snapshot.light !== null;
  if (id === 'terrain') return snapshot.terrain !== null;
  return false;
}

export function AddComponentPicker({ snapshot, projectSchemas, onAdd }: AddComponentPickerProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const frame = window.requestAnimationFrame(() => searchRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (event: PointerEvent) => {
      if (rootRef.current?.contains(event.target as Node)) return;
      setOpen(false);
      setQuery('');
    };
    document.addEventListener('pointerdown', handlePointerDown);
    return () => document.removeEventListener('pointerdown', handlePointerDown);
  }, [open]);

  const options = useMemo(() => {
    const attachedProjectTypes = new Set(snapshot.projectComponents.map((component) => component.typeId));
    const projectOptions = projectSchemas
      .filter((schema) => schema.projectComponent !== false)
      .filter((schema) => schema.allowMultiple === true || !attachedProjectTypes.has(schema.id))
      .map<AddComponentOption>((schema) => ({
        id: schema.id,
        label: schema.displayName,
        category: schema.category?.trim() || 'Project',
        tooltip: schema.tooltip,
      }));

    return [...builtInOptions.filter((option) => !builtInAlreadyPresent(snapshot, option.id)), ...projectOptions];
  }, [projectSchemas, snapshot]);

  const grouped = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const filtered = needle
      ? options.filter((option) =>
          `${option.label} ${option.category} ${option.tooltip ?? ''}`.toLocaleLowerCase().includes(needle),
        )
      : options;
    const groups = new Map<string, AddComponentOption[]>();
    filtered.forEach((option) => {
      const group = groups.get(option.category) ?? [];
      group.push(option);
      groups.set(option.category, group);
    });
    return [...groups.entries()];
  }, [options, query]);

  const add = async (option: AddComponentOption) => {
    if (!(await onAdd(option.id, option.label))) return;
    setOpen(false);
    setQuery('');
  };

  return (
    <div className="inspector-add-component-picker" ref={rootRef}>
      <button
        aria-expanded={open}
        className="inspector-add-component-trigger"
        onClick={() => setOpen((value) => !value)}
        type="button"
      >
        <Plus aria-hidden="true" size={15} />
        <span>Add Component</span>
      </button>
      {open && (
        <div className="inspector-add-component-popover" role="dialog" aria-label="Add Component">
          <label className="inspector-add-component-search">
            <Search aria-hidden="true" size={14} />
            <input
              ref={searchRef}
              aria-label="Search add components"
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key !== 'Escape') return;
                event.preventDefault();
                setOpen(false);
                setQuery('');
              }}
              placeholder="Search components"
              value={query}
            />
          </label>
          <div className="inspector-add-component-results">
            {grouped.map(([category, entries]) => (
              <section className="inspector-add-component-group" key={category}>
                <h4>{category}</h4>
                {entries.map((option) => (
                  <button key={option.id} onClick={() => void add(option)} title={option.tooltip} type="button">
                    {option.label}
                  </button>
                ))}
              </section>
            ))}
            {!grouped.length && <div className="inspector-add-component-empty">No components match “{query}”.</div>}
          </div>
        </div>
      )}
    </div>
  );
}
