import { UiTextInput, UiToggleButton } from '../ui';

type Props = {
  enabled: boolean;
  name: string;
  readOnly?: boolean;
  nameLabel?: string;
  onEnabledChange: (enabled: boolean) => void;
  onNameChange: (name: string) => void;
};

export function MaterialNodeParameterControl({
  enabled,
  name,
  readOnly = false,
  nameLabel = 'Parameter name',
  onEnabledChange,
  onNameChange,
}: Props) {
  return (
    <div className="material-node-parameter-toggle">
      <UiToggleButton
        checked={enabled}
        disabled={readOnly}
        label="Parameter"
        onCheckedChange={onEnabledChange}
      />
      <UiTextInput
        aria-label={nameLabel}
        disabled={readOnly || !enabled}
        value={name}
        onChange={(event) => onNameChange(event.target.value)}
      />
    </div>
  );
}
