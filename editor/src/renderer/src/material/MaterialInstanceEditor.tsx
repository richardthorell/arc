import { useMemo } from 'react';
import { ExternalLink, RotateCcw } from 'lucide-react';

import { openAssetEditorDocument } from '../editors/editorRegistry';
import type { EditorDocument } from '../editors/editorTypes';
import { UiButton, UiPanelCard, UiPanelCardRow, UiSelect } from '../ui';
import {
  selectedFunctionExtraParameters,
  type MaterialInstanceFunctionOption,
  type MaterialInstanceParentParameter,
} from './materialInstanceAuthoring';
import {
  replaceMaterialInstanceAsset,
  setMaterialInstanceParent,
  useMaterialInstanceDocumentState,
} from './materialInstanceDocumentState';

const valueAsNumbers = (value: unknown, count: number) => {
  if (typeof value === 'number') return [value];
  if (!Array.isArray(value)) return Array.from({ length: count }, () => 0);
  return Array.from({ length: count }, (_, index) =>
    typeof value[index] === 'number' && Number.isFinite(value[index]) ? value[index] : 0,
  );
};

function ValueEditor({
  value,
  type,
  disabled,
  onChange,
}: {
  value: unknown;
  type: string;
  disabled?: boolean;
  onChange: (value: unknown) => void;
}) {
  if (type === 'texture2d') {
    return (
      <input
        aria-label="Texture path"
        disabled={disabled}
        placeholder="Texture asset path"
        value={typeof value === 'string' ? value : ''}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  }
  const count = type === 'vec2' ? 2 : type === 'vec3' ? 3 : type === 'vec4' ? 4 : 1;
  const numbers = valueAsNumbers(value, count);
  if (count === 1) {
    return (
      <input
        aria-label="Parameter value"
        disabled={disabled}
        step="0.01"
        type="number"
        value={numbers[0]}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    );
  }
  return (
    <div className="material-instance-vector">
      {numbers.map((component, index) => (
        <input
          aria-label={`Component ${index + 1}`}
          disabled={disabled}
          key={index}
          step="0.01"
          type="number"
          value={component}
          onChange={(event) => {
            const next = [...numbers];
            next[index] = Number(event.target.value);
            onChange(next);
          }}
        />
      ))}
    </div>
  );
}

function ParameterRow({
  document,
  parameter,
}: {
  document: EditorDocument;
  parameter: MaterialInstanceParentParameter;
}) {
  const state = useMaterialInstanceDocumentState(document);
  const override = state.asset.parameterOverrides.find((entry) => entry.parameterId === parameter.id);
  const overridden = Boolean(override);
  const value = override?.value ?? parameter.value;

  const setOverride = (next: unknown) =>
    void replaceMaterialInstanceAsset(document, (asset) => ({
      ...asset,
      parameterOverrides: [
        ...asset.parameterOverrides.filter((entry) => entry.parameterId !== parameter.id),
        { parameterId: parameter.id, value: next },
      ],
    }));

  return (
    <UiPanelCardRow label={parameter.name}>
      <div className="material-instance-property">
        <input
          aria-label={`Override ${parameter.name}`}
          checked={overridden}
          disabled={document.readOnly}
          type="checkbox"
          onChange={(event) => {
            void replaceMaterialInstanceAsset(document, (asset) => ({
              ...asset,
              parameterOverrides: event.target.checked
                ? [
                    ...asset.parameterOverrides.filter((entry) => entry.parameterId !== parameter.id),
                    { parameterId: parameter.id, value: parameter.value },
                  ]
                : asset.parameterOverrides.filter((entry) => entry.parameterId !== parameter.id),
            }));
          }}
        />
        <ValueEditor
          disabled={document.readOnly || !overridden}
          type={parameter.type}
          value={value}
          onChange={setOverride}
        />
        {overridden && (
          <button
            aria-label={`Reset ${parameter.name}`}
            className="inspector-field-reset"
            disabled={document.readOnly}
            onClick={() =>
              void replaceMaterialInstanceAsset(document, (asset) => ({
                ...asset,
                parameterOverrides: asset.parameterOverrides.filter((entry) => entry.parameterId !== parameter.id),
              }))
            }
            title="Reset to parent"
            type="button"
          >
            <RotateCcw aria-hidden="true" size={12} />
          </button>
        )}
      </div>
    </UiPanelCardRow>
  );
}

export function MaterialInstanceEditor({ document }: { document: EditorDocument }) {
  const state = useMaterialInstanceDocumentState(document);
  const materialParents = useMemo(
    () => state.assets.filter((asset) => asset.kind === 'material' && asset.guid),
    [state.assets],
  );

  if (state.loading || !state.loaded)
    return <div className="editor-empty-state">{state.message || 'Loading Material Instance…'}</div>;

  const parentGuid = state.asset.parent.guid;
  return (
    <section
      className="material-editor"
      style={{ gridTemplateColumns: 'minmax(360px, 0.9fr) 5px minmax(480px, 1.1fr)' }}
    >
      <div className="material-editor-preview-region">
        <div className="material-instance-preview">
          {state.previewDataUrl ? (
            <img alt={`${state.asset.name} preview`} src={state.previewDataUrl} />
          ) : (
            <div className="editor-empty-state">
              {state.previewLoading ? 'Rendering preview…' : 'Preview unavailable'}
            </div>
          )}
        </div>
        {state.message && (
          <div className="material-editor-message" role="status">
            {state.message}
          </div>
        )}
      </div>

      <div className="material-editor-divider" aria-hidden="true" />

      <aside className="material-editor-sidebar editor-property-panel">
        <div className="material-settings-region">
          <UiPanelCard className="material-settings-card" title="Material Instance">
            <UiPanelCardRow label="Name">
              <input
                disabled={document.readOnly}
                value={state.asset.name}
                onChange={(event) =>
                  void replaceMaterialInstanceAsset(document, (asset) => ({ ...asset, name: event.target.value }))
                }
              />
            </UiPanelCardRow>
            <UiPanelCardRow label="Parent">
              <div className="material-instance-property">
                <UiSelect
                  ariaLabel="Parent Material"
                  disabled={document.readOnly}
                  options={materialParents.map((asset) => ({
                    value: asset.guid ?? '',
                    label: asset.title?.trim() || asset.name,
                  }))}
                  value={parentGuid}
                  onValueChange={(guid) => {
                    const parent = materialParents.find((asset) => asset.guid === guid);
                    if (parent) void setMaterialInstanceParent(document, parent);
                  }}
                />
                {state.parentModel && (
                  <UiButton
                    onClick={() => openAssetEditorDocument(state.parentModel!.asset)}
                    title="Open parent Material"
                  >
                    <ExternalLink aria-hidden="true" size={13} />
                  </UiButton>
                )}
              </div>
            </UiPanelCardRow>
          </UiPanelCard>

          <UiPanelCard className="material-settings-card" title="Parameters">
            {(state.parentModel?.parameters.length ?? 0) === 0 && (
              <p className="inspector-subsection-empty">No exposed parent parameters.</p>
            )}
            {state.parentModel?.parameters.map((parameter) => (
              <ParameterRow document={document} key={parameter.id} parameter={parameter} />
            ))}
          </UiPanelCard>

          <UiPanelCard className="material-settings-card" title="Function Overrides">
            {(state.parentModel?.functionSlots.length ?? 0) === 0 && (
              <p className="inspector-subsection-empty">No overridable Function Slots.</p>
            )}
            {state.parentModel?.functionSlots.map((slot) => {
              const authored = state.asset.functionOverrides.find((entry) => entry.slotId === slot.id);
              const selected =
                (authored
                  ? slot.compatibleFunctions.find((option) => option.reference.guid === authored.function.guid)
                  : slot.defaultFunction) ?? slot.defaultFunction;
              const defaultGuid = slot.defaultFunction?.reference.guid ?? '';
              const selectedGuid = selected?.reference.guid ?? defaultGuid;
              const setFunction = (option: MaterialInstanceFunctionOption | null) => {
                void replaceMaterialInstanceAsset(document, (asset) => ({
                  ...asset,
                  functionOverrides:
                    !option || option.reference.guid === defaultGuid
                      ? asset.functionOverrides.filter((entry) => entry.slotId !== slot.id)
                      : [
                          ...asset.functionOverrides.filter((entry) => entry.slotId !== slot.id),
                          { slotId: slot.id, function: option.reference, inputOverrides: [] },
                        ],
                }));
              };

              return (
                <div className="material-instance-function-slot" key={slot.id}>
                  <UiPanelCardRow label={slot.name}>
                    <UiSelect
                      ariaLabel={`${slot.name} function`}
                      disabled={document.readOnly}
                      options={slot.compatibleFunctions.map((option) => ({
                        value: option.reference.guid,
                        label: option.document.name,
                      }))}
                      value={selectedGuid}
                      onValueChange={(guid) =>
                        setFunction(slot.compatibleFunctions.find((option) => option.reference.guid === guid) ?? null)
                      }
                    />
                  </UiPanelCardRow>

                  {selected &&
                    selectedFunctionExtraParameters(slot, selected).map((parameter) => {
                      const functionOverride = state.asset.functionOverrides.find((entry) => entry.slotId === slot.id);
                      const inputOverride = functionOverride?.inputOverrides.find(
                        (entry) => entry.pinId === parameter.pinId,
                      );
                      const overridden = Boolean(inputOverride);
                      return (
                        <UiPanelCardRow key={parameter.id} label={parameter.name}>
                          <div className="material-instance-property">
                            <input
                              aria-label={`Override ${slot.name} ${parameter.name}`}
                              checked={overridden}
                              disabled={document.readOnly || !functionOverride}
                              type="checkbox"
                              onChange={(event) =>
                                void replaceMaterialInstanceAsset(document, (asset) => ({
                                  ...asset,
                                  functionOverrides: asset.functionOverrides.map((entry) =>
                                    entry.slotId !== slot.id
                                      ? entry
                                      : {
                                          ...entry,
                                          inputOverrides: event.target.checked
                                            ? [
                                                ...entry.inputOverrides.filter(
                                                  (input) => input.pinId !== parameter.pinId,
                                                ),
                                                { pinId: parameter.pinId, value: parameter.value },
                                              ]
                                            : entry.inputOverrides.filter((input) => input.pinId !== parameter.pinId),
                                        },
                                  ),
                                }))
                              }
                            />
                            <ValueEditor
                              disabled={document.readOnly || !functionOverride || !overridden}
                              type={parameter.type}
                              value={inputOverride?.value ?? parameter.value}
                              onChange={(value) =>
                                void replaceMaterialInstanceAsset(document, (asset) => ({
                                  ...asset,
                                  functionOverrides: asset.functionOverrides.map((entry) =>
                                    entry.slotId !== slot.id
                                      ? entry
                                      : {
                                          ...entry,
                                          inputOverrides: [
                                            ...entry.inputOverrides.filter((input) => input.pinId !== parameter.pinId),
                                            { pinId: parameter.pinId, value },
                                          ],
                                        },
                                  ),
                                }))
                              }
                            />
                          </div>
                        </UiPanelCardRow>
                      );
                    })}
                </div>
              );
            })}
          </UiPanelCard>
        </div>
      </aside>
    </section>
  );
}
