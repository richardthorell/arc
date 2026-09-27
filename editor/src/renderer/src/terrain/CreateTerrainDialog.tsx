import { useMemo, useState } from 'react';
import { Mountain, X } from 'lucide-react';

import type { HostEntityId, HostResponse } from '../inspector/inspectorTypes';
import { UiButton } from '../ui/UiButton';
import { UiNumericInput } from '../ui/UiNumericInput';
import { UiPropertyCard } from '../ui/UiPropertyCard';
import { UiSelect } from '../ui/UiSelect';

const resolutions = [257, 513, 1025, 2049, 4097] as const;
const sourceOptions = [
  { value: 'flat', label: 'Flat' },
  { value: 'procedural', label: 'Domain Warped' },
] as const;
const resolutionOptions = resolutions.map((value) => ({ value: String(value), label: `${value} x ${value}` }));
const patchOptions = [16, 32, 64].map((value) => ({ value: String(value), label: `${value} quads` }));

export function CreateTerrainDialog({
  parent,
  command,
  onClose,
  onCreated,
}: {
  parent?: HostEntityId;
  command: (type: string, payload: Record<string, unknown>) => Promise<HostResponse>;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [size, setSize] = useState(180);
  const [minimumElevation, setMinimumElevation] = useState(0);
  const [maximumElevation, setMaximumElevation] = useState(48);
  const [resolution, setResolution] = useState<(typeof resolutions)[number]>(257);
  const [patchQuads, setPatchQuads] = useState(32);
  const [source, setSource] = useState<'flat' | 'procedural'>('flat');
  const [seed, setSeed] = useState(1);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const estimate = useMemo(() => {
    const samples = resolution * resolution;
    return {
      cpu: (samples * 8) / (1024 * 1024),
      gpu: (samples * 8) / (1024 * 1024),
      history: (samples * 12) / (1024 * 1024),
    };
  }, [resolution]);

  const create = async () => {
    setBusy(true);
    setError('');
    const response = await command('terrain.create', {
      size,
      minimumElevation,
      maximumElevation,
      resolution,
      patchQuads,
      source,
      seed,
      ...(parent ? { parent } : {}),
    });
    setBusy(false);
    if (!response.succeeded) return setError(response.error || 'Terrain creation failed');
    onCreated();
    onClose();
  };

  const fields = [
    {
      id: 'source',
      label: 'Source',
      control: (
        <UiSelect
          ariaLabel="Source"
          options={sourceOptions}
          value={source}
          onValueChange={(value) => setSource(value as typeof source)}
        />
      ),
    },
    {
      id: 'size',
      label: 'Physical Size (m)',
      control: (
        <UiNumericInput
          ariaLabel="Physical Size (m)"
          max={262144}
          min={1}
          precision={0}
          scrubSensitivity={1}
          step={1}
          value={size}
          onCommit={setSize}
        />
      ),
    },
    {
      id: 'minimumElevation',
      label: 'Minimum Elevation (m)',
      control: (
        <UiNumericInput
          ariaLabel="Minimum Elevation (m)"
          precision={1}
          scrubSensitivity={0.5}
          step={1}
          value={minimumElevation}
          onCommit={setMinimumElevation}
        />
      ),
    },
    {
      id: 'maximumElevation',
      label: 'Maximum Elevation (m)',
      control: (
        <UiNumericInput
          ariaLabel="Maximum Elevation (m)"
          precision={1}
          scrubSensitivity={0.5}
          step={1}
          value={maximumElevation}
          onCommit={setMaximumElevation}
        />
      ),
    },
    {
      id: 'resolution',
      label: 'Resolution',
      control: (
        <UiSelect
          ariaLabel="Resolution"
          options={resolutionOptions}
          value={String(resolution)}
          onValueChange={(value) => setResolution(Number(value) as typeof resolution)}
        />
      ),
    },
    {
      id: 'patchQuads',
      label: 'Patch Topology',
      control: (
        <UiSelect
          ariaLabel="Patch Topology"
          options={patchOptions}
          value={String(patchQuads)}
          onValueChange={(value) => setPatchQuads(Number(value))}
        />
      ),
    },
    ...(source === 'procedural'
      ? [
          {
            id: 'seed',
            label: 'Seed',
            control: (
              <UiNumericInput
                ariaLabel="Seed"
                min={0}
                precision={0}
                scrubSensitivity={1}
                step={1}
                value={seed}
                onCommit={setSeed}
              />
            ),
          },
        ]
      : []),
  ];

  return (
    <div className="modal-backdrop" role="presentation">
      <section aria-label="Create terrain" aria-modal="true" className="terrain-create-dialog" role="dialog">
        <header>
          <span>
            <Mountain size={18} /> Create Terrain
          </span>
          <UiButton aria-label="Close" variant="icon" onClick={onClose}>
            <X size={16} />
          </UiButton>
        </header>
        <UiPropertyCard className="terrain-create-fields" expandable={false} fields={fields} title="Terrain" />
        <div className="terrain-memory-estimate">
          <span>CPU {estimate.cpu.toFixed(1)} MiB</span>
          <span>GPU {estimate.gpu.toFixed(1)} MiB</span>
          <span>Undo {estimate.history.toFixed(1)} MiB</span>
        </div>
        {estimate.history > 64 && (
          <p className="terrain-operation-warning">This terrain exceeds the 64 MiB undo budget and will be rejected.</p>
        )}
        {error && <p className="command-error">{error}</p>}
        <footer>
          <UiButton onClick={onClose}>Cancel</UiButton>
          <UiButton variant="primary" disabled={busy || estimate.history > 64} onClick={() => void create()}>
            {busy ? 'Creating...' : 'Create Terrain'}
          </UiButton>
        </footer>
      </section>
    </div>
  );
}
