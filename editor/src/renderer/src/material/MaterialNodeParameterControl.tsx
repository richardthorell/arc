import { UiTextInput, UiToggleButton } from '../ui';

type Props = {
  enabled: boolean;
  name: string;
  group?: string;
  order?: number;
  readOnly?: boolean;
  nameLabel?: string;
  onEnabledChange: (enabled: boolean) => void;
  onNameChange: (name: string) => void;
  onGroupChange?: (group: string) => void;
  onOrderChange?: (order: number | undefined) => void;
};

export function MaterialNodeParameterControl({
  enabled,
  name,
  group = '',
  order,
  readOnly = false,
  nameLabel = 'Parameter name',
  onEnabledChange,
  onNameChange,
  onGroupChange,
  onOrderChange,
}: Props) {
  return (
    <div className="material-node-parameter-toggle">
      <div className="material-node-parameter-main">
        <UiToggleButton checked={enabled} disabled={readOnly} label="Parameter" onCheckedChange={onEnabledChange} />
        <UiTextInput
          aria-label={nameLabel}
          disabled={readOnly || !enabled}
          value={name}
          onChange={(event) => onNameChange(event.target.value)}
        />
      </div>
      {enabled && onGroupChange && (
        <div className="material-node-parameter-metadata">
          <UiTextInput
            aria-label="Parameter group"
            disabled={readOnly}
            placeholder="Group"
            value={group}
            onChange={(event) => onGroupChange(event.target.value)}
          />
          {onOrderChange && (
            <UiTextInput
              aria-label="Parameter order"
              disabled={readOnly}
              inputMode="numeric"
              placeholder="Order"
              value={order === undefined ? '' : String(order)}
              onChange={(event) => {
                const value = event.target.value.trim();
                if (!value) onOrderChange(undefined);
                else {
                  const parsed = Number(value);
                  if (Number.isFinite(parsed)) onOrderChange(parsed);
                }
              }}
            />
          )}
        </div>
      )}
    </div>
  );
}
