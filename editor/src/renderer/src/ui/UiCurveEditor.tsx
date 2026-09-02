import { useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { RotateCcw, Trash2 } from 'lucide-react';
import { UiIconButton } from './UiIconButton';
import './UiCurveEditor.css';

export type UiCurveInterpolation = 'constant' | 'linear' | 'smooth' | 'manual';
export type UiCurvePoint = {
  x: number;
  y: number;
  inTangent: number;
  outTangent: number;
  interpolation: UiCurveInterpolation;
};
export type UiCurveHistogram = number[];
export type UiCurveEditorProps = {
  value: UiCurvePoint[];
  onChange: (value: UiCurvePoint[]) => void;
  histogram?: UiCurveHistogram;
  disabled?: boolean;
  ariaLabel?: string;
};
const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const identity = (): UiCurvePoint[] => [
  { x: 0, y: 0, inTangent: 1, outTangent: 1, interpolation: 'smooth' },
  { x: 1, y: 1, inTangent: 1, outTangent: 1, interpolation: 'smooth' },
];
const slope = (a: UiCurvePoint, b: UiCurvePoint) => (b.x > a.x ? (b.y - a.y) / (b.x - a.x) : 0);
const autoTangent = (points: UiCurvePoint[], i: number) => {
  if (i === 0) return slope(points[0], points[1]);
  if (i === points.length - 1) return slope(points[i - 1], points[i]);
  const l = slope(points[i - 1], points[i]);
  const r = slope(points[i], points[i + 1]);
  return l * r <= 0 ? 0 : (l + r) / 2;
};
export function evaluateUiCurve(points: UiCurvePoint[], input: number) {
  const x = clamp(input);
  if (points.length < 2) return x;
  let ri = points.findIndex((p) => p.x > x);
  if (ri < 0) return points.at(-1)!.y;
  if (ri === 0) return points[0].y;
  const li = ri - 1;
  const a = points[li],
    b = points[ri];
  const t = (x - a.x) / Math.max(1e-6, b.x - a.x);
  if (a.interpolation === 'constant') return a.y;
  if (a.interpolation === 'linear') return a.y + (b.y - a.y) * t;
  const m0 = a.interpolation === 'manual' ? a.outTangent : autoTangent(points, li);
  const m1 = b.interpolation === 'manual' ? b.inTangent : autoTangent(points, ri);
  const t2 = t * t,
    t3 = t2 * t,
    w = b.x - a.x;
  return clamp(
    (2 * t3 - 3 * t2 + 1) * a.y + (t3 - 2 * t2 + t) * w * m0 + (-2 * t3 + 3 * t2) * b.y + (t3 - t2) * w * m1,
  );
}
export function UiCurveEditor({
  value,
  onChange,
  histogram,
  disabled,
  ariaLabel = 'Curve editor',
}: UiCurveEditorProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const path = useMemo(
    () =>
      Array.from({ length: 129 }, (_, i) => {
        const x = i / 128;
        const y = evaluateUiCurve(value, x);
        return `${i ? 'L' : 'M'} ${x * 100} ${(1 - y) * 100}`;
      }).join(' '),
    [value],
  );
  const histogramPath = useMemo(() => {
    if (!histogram?.length) return '';
    const peak = Math.max(1, ...histogram);
    return (
      histogram
        .map((v, i) => `${i ? 'L' : 'M'} ${(i / (histogram.length - 1)) * 100} ${100 - (v / peak) * 92}`)
        .join(' ') + ' L 100 100 L 0 100 Z'
    );
  }, [histogram]);
  const eventPoint = (event: ReactPointerEvent<SVGSVGElement>) => {
    const r = event.currentTarget.getBoundingClientRect();
    return { x: clamp((event.clientX - r.left) / r.width), y: clamp(1 - (event.clientY - r.top) / r.height) };
  };
  const move = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (dragging === null || disabled) return;
    const p = eventPoint(event);
    const next = value.map((point, i) =>
      i === dragging
        ? {
            ...point,
            x:
              i === 0
                ? 0
                : i === value.length - 1
                  ? 1
                  : Math.min(value[i + 1].x - 0.005, Math.max(value[i - 1].x + 0.005, p.x)),
            y: p.y,
          }
        : point,
    );
    onChange(next);
  };
  const add = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (disabled || value.length >= 32) return;
    const p = eventPoint(event);
    const next = [...value, { x: p.x, y: p.y, inTangent: 1, outTangent: 1, interpolation: 'smooth' as const }].sort(
      (a, b) => a.x - b.x,
    );
    onChange(next);
    setSelected(next.findIndex((q) => q.x === p.x && q.y === p.y));
  };
  const remove = () => {
    if (selected === null || selected === 0 || selected === value.length - 1) return;
    onChange(value.filter((_, i) => i !== selected));
    setSelected(null);
  };
  return (
    <div className="ui-curve-editor" aria-label={ariaLabel}>
      <div className="ui-curve-toolbar">
        <select
          aria-label="Curve interpolation"
          disabled={disabled || selected === null}
          value={selected === null ? 'smooth' : value[selected].interpolation}
          onChange={(e) =>
            selected !== null &&
            onChange(
              value.map((p, i) =>
                i === selected ? { ...p, interpolation: e.target.value as UiCurveInterpolation } : p,
              ),
            )
          }
        >
          <option value="constant">Constant</option>
          <option value="linear">Linear</option>
          <option value="smooth">Smooth</option>
          <option value="manual">Manual</option>
        </select>
        <UiIconButton
          label="Delete curve point"
          aria-label="Delete curve point"
          disabled={disabled || selected === null || selected === 0 || selected === value.length - 1}
          onClick={remove}
        >
          <Trash2 size={13} />
        </UiIconButton>
        <UiIconButton
          label="Reset curve"
          aria-label="Reset curve"
          disabled={disabled}
          onClick={() => {
            onChange(identity());
            setSelected(null);
          }}
        >
          <RotateCcw size={13} />
        </UiIconButton>
      </div>
      <svg
        ref={svgRef}
        className="ui-curve-canvas"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        onDoubleClick={add}
        onPointerMove={move}
        onPointerUp={(e) => {
          setDragging(null);
          if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
        }}
      >
        <g className="ui-curve-grid">
          <path d="M 0 25 H 100 M 0 50 H 100 M 0 75 H 100 M 25 0 V 100 M 50 0 V 100 M 75 0 V 100" />
        </g>
        {histogramPath && <path className="ui-curve-histogram" d={histogramPath} />}
        <path className="ui-curve-line" d={path} />
        {value.map((p, i) => (
          <circle
            aria-label={`Curve point ${i + 1}`}
            className={selected === i ? 'ui-curve-point selected' : 'ui-curve-point'}
            cx={p.x * 100}
            cy={(1 - p.y) * 100}
            key={`${i}-${p.x}`}
            r="2.2"
            onPointerDown={(e) => {
              if (disabled) return;
              e.stopPropagation();
              setSelected(i);
              setDragging(i);
              e.currentTarget.ownerSVGElement?.setPointerCapture(e.pointerId);
            }}
          />
        ))}
      </svg>
      <div className="ui-curve-readout">
        {selected === null
          ? 'Double-click to add a point'
          : `X ${value[selected].x.toFixed(3)}  Y ${value[selected].y.toFixed(3)}`}
      </div>
    </div>
  );
}
