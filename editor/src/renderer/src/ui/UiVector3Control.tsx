import { Link2, RotateCcw } from 'lucide-react';

import { UiIconButton } from './UiIconButton';
import { UiNumericInput } from './UiNumericInput';

import './UiVector3Control.css';

export type UiVector3Axis = 'x' | 'y' | 'z';

export type UiVector3Value = {
  x: number;
  y: number;
  z: number;
};

export type UiVector3ControlProps = {
  label: string;
  value: UiVector3Value;
  precision: number;
  step: number;
  scrubSensitivity: number;
  unit?: string;
  tooltip?: string;
  linkable?: boolean;
  linked?: boolean;
  mixed?: boolean;
  showLabel?: boolean;
  showActions?: boolean;
  onToggleLinked?: () => void;
  onReset?: () => void;
  onPreview: (axis: UiVector3Axis, value: number) => void;
  onCommit: (axis: UiVector3Axis, value: number) => void;
};

export function UiVector3Control({
  label,
  value,
  precision,
  step,
  scrubSensitivity,
  unit,
  tooltip,
  linkable = false,
  linked = false,
  mixed = false,
  showLabel = true,
  showActions = true,
  onToggleLinked,
  onReset,
  onPreview,
  onCommit,
}: UiVector3ControlProps) {
  return (
    <div className={['ui-vector3-control', showLabel ? 'is-labeled' : ''].filter(Boolean).join(' ')} title={tooltip}>
      {showLabel && <span className="ui-vector3-label">{label}</span>}
      <div className={['ui-vector3-value', !showActions ? 'has-no-actions' : ''].filter(Boolean).join(' ')}>
        <div className="ui-vector3-axis-grid">
          {(['x', 'y', 'z'] as const).map((axis) => (
            <UiNumericInput
              ariaLabel={`${label} ${axis.toUpperCase()}`}
              key={axis}
              mixed={mixed}
              precision={precision}
              scrubClassName={`axis-${axis}`}
              scrubLabel={axis.toUpperCase()}
              scrubSensitivity={scrubSensitivity}
              step={step}
              unit={unit}
              value={value[axis]}
              onCommit={(next) => onCommit(axis, next)}
              onPreview={(next) => onPreview(axis, next)}
            />
          ))}
        </div>
        {showActions && (
          <div className="ui-vector3-actions" aria-hidden={!linkable && !onReset ? true : undefined}>
            <span className="ui-vector3-action-slot">
              {linkable && (
                <UiIconButton
                  className={['ui-vector3-action', 'ui-vector3-link', linked ? 'is-linked' : '']
                    .filter(Boolean)
                    .join(' ')}
                  label={`${linked ? 'Unlink' : 'Link'} ${label.toLocaleLowerCase()} axes`}
                  onClick={onToggleLinked}
                  title={`${linked ? 'Unlink' : 'Link'} ${label.toLocaleLowerCase()} axes`}
                  type="button"
                >
                  <Link2 aria-hidden="true" size={13} strokeWidth={2} />
                </UiIconButton>
              )}
            </span>
            <span className="ui-vector3-action-slot">
              {onReset && (
                <UiIconButton
                  className="ui-vector3-action ui-vector3-reset"
                  label={`Reset ${label}`}
                  onClick={onReset}
                  title={`Reset ${label}`}
                  type="button"
                >
                  <RotateCcw aria-hidden="true" size={12} />
                </UiIconButton>
              )}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
