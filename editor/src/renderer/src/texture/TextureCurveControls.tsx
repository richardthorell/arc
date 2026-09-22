import { useState } from 'react';

import { UiButton, UiCurveEditor, UiPropertyCard, UiToggleButton, type UiCurveHistogram } from '../ui';
import type { TextureCurve, TextureSettingsPatch, TextureSettingsSnapshot } from './textureSettings';

type CurveKey = 'curveMaster' | 'curveR' | 'curveG' | 'curveB' | 'curveA';

const channels: Array<[CurveKey, string]> = [
  ['curveMaster', 'Master'],
  ['curveR', 'R'],
  ['curveG', 'G'],
  ['curveB', 'B'],
  ['curveA', 'A'],
];

export function TextureCurveControls({
  draft,
  histogram,
  update,
}: {
  draft: TextureSettingsSnapshot;
  histogram?: { r: number[]; g: number[]; b: number[]; a: number[] };
  update: (patch: TextureSettingsPatch) => void;
}) {
  const [active, setActive] = useState<CurveKey>('curveMaster');
  const [collapsed, setCollapsed] = useState(true);

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
    <UiPropertyCard
      className="texture-inspector-section"
      collapsed={collapsed}
      fields={[
        {
          id: 'enable-curves',
          label: 'Enable Curves',
          control: (
            <UiToggleButton
              aria-label="Enable Curves"
              checked={draft.curvesEnabled}
              onCheckedChange={(curvesEnabled) => update({ curvesEnabled })}
            />
          ),
        },
        {
          id: 'curve-editor',
          fullWidth: true,
          control: (
            <div className="texture-curve-editor-field">
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
            </div>
          ),
        },
      ]}
      onToggle={() => setCollapsed((value) => !value)}
      title="Curves"
    />
  );
}
