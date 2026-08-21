import { useMemo, useState } from 'react';
import { Mountain, X } from 'lucide-react';

import type { HostEntityId, HostResponse } from '../inspector/inspectorTypes';

const resolutions = [257, 513, 1025, 2049, 4097] as const;

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

  return (
    <div className="modal-backdrop" role="presentation">
      <section aria-label="Create terrain" aria-modal="true" className="terrain-create-dialog" role="dialog">
        <header>
          <span>
            <Mountain size={18} /> Create Terrain
          </span>
          <button aria-label="Close" onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div className="terrain-create-fields">
          <label>
            Source
            <select value={source} onChange={(event) => setSource(event.target.value as typeof source)}>
              <option value="flat">Flat</option>
              <option value="procedural">Domain Warped</option>
            </select>
          </label>
          <label>
            Physical Size (m)
            <input
              type="number"
              min={1}
              max={262144}
              value={size}
              onChange={(event) => setSize(Number(event.target.value))}
            />
          </label>
          <label>
            Minimum Elevation (m)
            <input
              type="number"
              value={minimumElevation}
              onChange={(event) => setMinimumElevation(Number(event.target.value))}
            />
          </label>
          <label>
            Maximum Elevation (m)
            <input
              type="number"
              value={maximumElevation}
              onChange={(event) => setMaximumElevation(Number(event.target.value))}
            />
          </label>
          <label>
            Resolution
            <select
              value={resolution}
              onChange={(event) => setResolution(Number(event.target.value) as typeof resolution)}
            >
              {resolutions.map((value) => (
                <option key={value} value={value}>
                  {value} x {value}
                </option>
              ))}
            </select>
          </label>
          <label>
            Patch Topology
            <select value={patchQuads} onChange={(event) => setPatchQuads(Number(event.target.value))}>
              {[16, 32, 64].map((value) => (
                <option key={value} value={value}>
                  {value} quads
                </option>
              ))}
            </select>
          </label>
          {source === 'procedural' && (
            <label>
              Seed
              <input type="number" min={0} value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </label>
          )}
        </div>
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
          <button onClick={onClose}>Cancel</button>
          <button disabled={busy || estimate.history > 64} onClick={() => void create()}>
            {busy ? 'Creating...' : 'Create Terrain'}
          </button>
        </footer>
      </section>
    </div>
  );
}
