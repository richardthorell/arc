import { useEffect, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import { materialDescribeOutput, type MaterialDescription } from '../../../common/materialOperationContract';
import { UiPanelCardRow, UiSelect, UiNumericInput, UiColorControl, UiSlider } from '../ui';
import { TexturePicker, type AssetThumbnailProvider } from './AssetPicker';
import './inspectorPolish.css';

type Asset = {
  id: string;
  guid?: string;
  typeId?: string;
  name: string;
  path: string;
  sourcePath?: string;
  kind: string;
  status: 'unknown' | 'queued' | 'ready' | 'dirty' | 'source' | 'stale' | 'importing' | 'failed' | 'missing';
  scope?: 'builtin' | 'project' | 'user' | 'organization' | 'procedural';
  readOnly?: boolean;
};
type Response = { succeeded: boolean; error?: string; payload?: unknown; sceneRevision: number; worldEpoch: number };
type Inspection = { description: MaterialDescription; sceneRevision: number; worldEpoch: number };

export async function inspectEntityMaterial(entityGuid: string): Promise<Inspection> {
  const response = (await window.arc.host.query('materials.describe', { entityGuid })) as Response;
  if (!response.succeeded) throw new Error(response.error || 'Material inspection failed');
  return {
    description: materialDescribeOutput.parse(response.payload),
    sceneRevision: response.sceneRevision,
    worldEpoch: response.worldEpoch,
  };
}

/** Both Inspector and agent mutations use the same revision-checked native material service. */
export async function editInspectedMaterial(
  inspection: Inspection,
  operation: string,
  value: Record<string, unknown>,
): Promise<void> {
  if (!inspection.description.entity) throw new Error('No entity material binding was inspected');
  const response = (await window.arc.host.command('materials.edit', {
    ...value,
    operation,
    entityGuid: inspection.description.entity.guid,
    definitionRevision: inspection.description.definitionRevision,
    expectedSceneRevision: inspection.sceneRevision,
    worldEpoch: inspection.worldEpoch,
  })) as Response;
  if (!response.succeeded) throw new Error(response.error || 'Material update failed; previous values retained');
}

export function MaterialParameterSubsection({
  assets,
  mixed = false,
  thumbnailProvider,
  value,
}: {
  assets: ReadonlyArray<Asset>;
  mixed?: boolean;
  referenceMode?: 'path' | 'guid';
  thumbnailProvider?: AssetThumbnailProvider;
  value: string;
}) {
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const generation = useRef(0);
  const pending = useRef(false);
  useEffect(
    () =>
      window.arc?.host?.onEvent?.((event) => {
        if (
          [
            'component.changed',
            'entity.selected',
            'scene.changed',
            'project.opened',
            'project.closed',
            'asset.changed',
          ].includes(event.type)
        )
          setRefresh((current) => current + 1);
      }),
    [],
  );
  useEffect(() => {
    const current = ++generation.current;
    setInspection(null);
    setError('');
    if (!value || mixed || value.startsWith('__arc_')) return;
    void (async () => {
      try {
        const selected = (await window.arc.host.query('entity.selected')) as Response;
        const payload = selected.payload as { guid?: string; selectionCount?: number } | undefined;
        if (!selected.succeeded || !payload?.guid)
          throw new Error(selected.error || 'Select an entity with a registered material');
        if ((payload.selectionCount ?? 1) > 1)
          throw new Error('Select one entity to edit its local material overrides');
        const result = await inspectEntityMaterial(payload.guid);
        if (current === generation.current) setInspection(result);
      } catch (cause) {
        if (current === generation.current) setError(cause instanceof Error ? cause.message : String(cause));
      }
    })();
    return () => {
      generation.current = current + 1;
    };
  }, [value, mixed, refresh]);

  const mutate = async (operation: string, fields: Record<string, unknown>) => {
    if (!inspection?.description.entity || pending.current) return;
    const current = generation.current;
    pending.current = true;
    setBusy(true);
    setError('');
    try {
      // Never retarget to a newly selected entity after a control was displayed.
      await editInspectedMaterial(inspection, operation, fields);
      const result = await inspectEntityMaterial(inspection.description.entity.guid);
      if (current === generation.current) setInspection(result);
    } catch (cause) {
      if (current === generation.current) setError(cause instanceof Error ? cause.message : String(cause));
      // A rejected stale edit should not leave the controls permanently bound to old authority.
      try {
        const result = await inspectEntityMaterial(inspection.description.entity.guid);
        if (current === generation.current) setInspection(result);
      } catch {
        /* Preserve the original actionable edit error. */
      }
    } finally {
      pending.current = false;
      setBusy(false);
    }
  };
  if (!value || mixed || value.startsWith('__arc_')) return null;
  const description = inspection?.description;

  // Native parameter IDs and slot IDs are authoritative: do not reparse the saved
  // graph. A function's exposed parameters are nested below its selector.
  const renderParameter = (parameter: MaterialDescription['parameters'][number]) => {
    const values = Array.isArray(parameter.resolvedValue)
      ? parameter.resolvedValue
      : typeof parameter.resolvedValue === 'number'
        ? [parameter.resolvedValue]
        : [];
    const texture =
      parameter.resolvedValue && typeof parameter.resolvedValue === 'object' && !Array.isArray(parameter.resolvedValue)
        ? parameter.resolvedValue.guid
        : '';
    const changeNumber = (index: number, next: number) => {
      if (!Number.isFinite(next) || next === values[index]) return;
      const updated = [...values];
      updated[index] = next;
      void mutate('setOverride', { parameterId: parameter.id, value: updated });
    };
    const isColor = parameter.semantic === 'baseColor' && values.length >= 3;
    const isScalar = parameter.type === 'float' && values.length === 1;
    return (
      <UiPanelCardRow
        className="inspector-material-parameter"
        key={parameter.id}
        label={parameter.name}
        align={parameter.type === 'texture2d' ? 'start' : 'center'}
      >
        <div className="inspector-material-parameter-control">
          {parameter.type === 'texture2d' ? (
            <TexturePicker
              allowEmpty
              assets={assets}
              label={parameter.name}
              showLabel={false}
              referenceMode="guid"
              thumbnailProvider={thumbnailProvider}
              value={texture}
              onChange={(_, asset) => {
                if (asset?.guid)
                  void mutate('setOverride', {
                    parameterId: parameter.id,
                    value: { guid: asset.guid, pathHint: asset.path },
                  });
              }}
            />
          ) : isColor ? (
            <UiColorControl
              label={parameter.name}
              allowAlpha={values.length === 4}
              value={{ x: values[0], y: values[1], z: values[2], w: values[3] ?? 1 }}
              onCommit={(color) =>
                void mutate('setOverride', {
                  parameterId: parameter.id,
                  value: [color.x, color.y, color.z, color.w].slice(0, values.length),
                })
              }
            />
          ) : (
            <div className="inspector-material-parameter-editors">
              <div className="inspector-material-parameter-values">
                {values.map((number, i) => (
                  <UiNumericInput
                    key={i}
                    precision={3}
                    step={0.01}
                    scrubSensitivity={0.005}
                    scrubLabel={values.length > 1 ? ['X', 'Y', 'Z', 'W'][i] : undefined}
                    scrubClassName={values.length > 1 ? `axis-${['x', 'y', 'z', 'w'][i]}` : undefined}
                    ariaLabel={values.length === 1 ? parameter.name : `${parameter.name} ${['X', 'Y', 'Z', 'W'][i]}`}
                    value={number}
                    min={parameter.range?.min}
                    max={parameter.range?.max}
                    onCommit={(next) => changeNumber(i, next)}
                  />
                ))}
              </div>
              {isScalar && parameter.range && (
                <UiSlider
                  aria-label={`${parameter.name} slider`}
                  min={parameter.range.min}
                  max={parameter.range.max}
                  step={Math.max(0.001, (parameter.range.max - parameter.range.min) / 100)}
                  value={values[0]}
                  onValueChange={(next) => changeNumber(0, next)}
                />
              )}
            </div>
          )}
          {parameter.valueStatus === 'unavailable' && <small>Value unavailable</small>}
          {parameter.overridden && (
            <button
              type="button"
              className="inspector-field-reset"
              aria-label={`Reset ${parameter.name}`}
              onClick={() => void mutate('resetOverride', { parameterId: parameter.id })}
            >
              <RotateCcw size={12} />
            </button>
          )}
        </div>
      </UiPanelCardRow>
    );
  };

  return (
    <section className="inspector-material-parameters" aria-label="Material Parameters">
      <h4>Material Parameters</h4>
      {error && <div role="alert">{error}</div>}
      {!description && !error && <span>Loading…</span>}
      {description?.diagnostics.map((diagnostic, i) => (
        <div role="status" key={i}>{diagnostic.message}</div>
      ))}
      <fieldset disabled={busy || !description?.definitionReady} style={{ border: 0, margin: 0, padding: 0 }}>
        {description?.functionSlots.map((slot) => {
          const children = description.parameters.filter((parameter) => parameter.slotId === slot.id);
          // Internal one-choice functions should not display fake selection controls,
          // but their exposed parameters must still be editable if present.
          if (slot.alternatives.length <= 1 && children.length === 0) return null;
          return (
            <div className="inspector-material-function" key={slot.id}>
              {slot.alternatives.length > 1 && (
                <UiPanelCardRow label={slot.name} className="inspector-material-function-selector">
                  <UiSelect
                    ariaLabel={`${slot.name} function`}
                    value={slot.selectedFunctionGuid}
                    options={slot.alternatives.map((entry) => ({
                      value: entry.guid,
                      label: assets.find((asset) => asset.guid === entry.guid)?.name ?? entry.pathHint,
                    }))}
                    onValueChange={(assetGuid) => void mutate('selectFunction', { slotId: slot.id, assetGuid })}
                  />
                </UiPanelCardRow>
              )}
              {slot.alternativesTruncated && (
                <small>Additional alternatives were omitted by the inspection budget.</small>
              )}
              {children.length > 0 && (
                <div className="inspector-material-function-parameters">
                  {children.map(renderParameter)}
                </div>
              )}
            </div>
          );
        })}
        <div className="inspector-material-parameter-list">
          {description?.parameters.filter((parameter) =>
            !parameter.slotId || !description.functionSlots.some((slot) => slot.id === parameter.slotId)
          ).map(renderParameter)}
        </div>
        {description && description.parameters.length === 0 && description.functionSlots.length === 0 && (
          <p className="inspector-subsection-empty">No exported parameters.</p>
        )}
      </fieldset>
    </section>
  );
}
