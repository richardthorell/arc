import { useEffect, useRef, useState } from 'react';
import type { AssetItem } from '../services/editorHostTypes';
import { UiButton, UiCurveEditor, UiPanelSection, type UiCurveHistogram } from '../ui';
import {
  patchTextureSettings,
  type TextureCurve,
  type TextureSettingsPatch,
  type TextureSettingsSnapshot,
} from './textureSettings';
import { useTextureSettings } from './useTextureSettings';
type CurveKey = 'curveMaster' | 'curveR' | 'curveG' | 'curveB' | 'curveA';
const channels: Array<[CurveKey, string]> = [
  ['curveMaster', 'Master'],
  ['curveR', 'R'],
  ['curveG', 'G'],
  ['curveB', 'B'],
  ['curveA', 'A'],
];
export function TextureCurveControls({
  asset,
  histogram,
}: {
  asset: AssetItem;
  histogram?: { r: number[]; g: number[]; b: number[]; a: number[] };
}) {
  const { settings } = useTextureSettings(asset.guid, asset.generation);
  const [draft, setDraft] = useState<TextureSettingsSnapshot | null>(settings);
  const [active, setActive] = useState<CurveKey>('curveMaster');
  const timer = useRef<number | null>(null);
  useEffect(() => setDraft(settings), [settings]);
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  if (!asset.guid || !draft || asset.readOnly) return null;
  const update = (patch: TextureSettingsPatch) => {
    setDraft((c) => (c ? { ...c, ...patch } : c));
    window.dispatchEvent(new CustomEvent('arc:texture-settings-preview', { detail: { guid: asset.guid, patch } }));
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => void patchTextureSettings(asset.guid!, patch), 250);
  };
  const h: UiCurveHistogram | undefined =
    active === 'curveR'
      ? histogram?.r
      : active === 'curveG'
        ? histogram?.g
        : active === 'curveB'
          ? histogram?.b
          : active === 'curveA'
            ? histogram?.a
            : histogram
              ? histogram.r.map((v, i) => v + histogram.g[i] + histogram.b[i])
              : undefined;
  return (
    <UiPanelSection className="texture-inspector-section" collapsed title="Curves">
      <label className="inspector-property texture-inspector-property">
        <span className="inspector-property-label">Enable Curves</span>
        <input
          checked={draft.curvesEnabled}
          type="checkbox"
          onChange={(e) => update({ curvesEnabled: e.target.checked })}
        />
      </label>
      <div className="texture-curve-tabs">
        {channels.map(([key, label]) => (
          <UiButton active={active === key} key={key} onClick={() => setActive(key)} variant="toolbar">
            {label}
          </UiButton>
        ))}
      </div>
      <UiCurveEditor
        ariaLabel={`${channels.find(([key]) => key === active)?.[1]} texture curve`}
        disabled={!draft.curvesEnabled || draft.semantic === 'normal'}
        histogram={h}
        value={draft[active]}
        onChange={(value) => update({ [active]: value as TextureCurve })}
      />
      {draft.semantic === 'normal' && (
        <div className="texture-stage3-note">RGB curves are bypassed for normal-map semantics.</div>
      )}
    </UiPanelSection>
  );
}
