import { ColorControl } from '../inspector/InspectorControls';

type UiColorValue = [number, number, number, number];

const toInspectorColor = ([x, y, z, w]: UiColorValue) => ({ x, y, z, w });
const fromInspectorColor = ({ x, y, z, w }: { x: number; y: number; z: number; w: number }): UiColorValue => [x, y, z, w];

export function UiColorControl({
  label,
  value,
  allowAlpha = true,
  onPreview,
  onCommit,
}: {
  label: string;
  value: UiColorValue;
  allowAlpha?: boolean;
  onPreview?: (value: UiColorValue) => void;
  onCommit: (value: UiColorValue) => void;
}) {
  return (
    <ColorControl
      label={label}
      showAlpha={allowAlpha}
      value={toInspectorColor(value)}
      onPreview={(next) => onPreview?.(fromInspectorColor(next))}
      onCommit={(next) => onCommit(fromInspectorColor(next))}
    />
  );
}
