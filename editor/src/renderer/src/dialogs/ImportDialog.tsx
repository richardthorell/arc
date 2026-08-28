import { FileImage, Upload } from 'lucide-react';
import { useState } from 'react';

import { UiButton, UiDialog } from '../ui';

import './ImportDialog.css';

type ImportDialogProps = {
  onClose?: () => void;
  onImport?: () => void;
  preview?: boolean;
};

export function ImportDialog({ onClose, onImport, preview = false }: ImportDialogProps) {
  const [destination, setDestination] = useState('Content/Textures');
  const [generateMipmaps, setGenerateMipmaps] = useState(true);
  const [srgb, setSrgb] = useState(true);
  const [compression, setCompression] = useState('Default (BC7)');

  return (
    <UiDialog
      className="import-dialog"
      footer={
        <>
          <UiButton onClick={onClose}>Cancel</UiButton>
          <UiButton onClick={onImport} variant="primary">
            Import 3 Assets
          </UiButton>
        </>
      }
      icon={<Upload aria-hidden="true" size={18} />}
      onClose={onClose}
      preview={preview}
      subtitle="Review files and import settings"
      title="Import Assets"
      width={620}
    >
      <div className="import-dialog-files">
        <header>
          <strong>Source files</strong>
          <span>3 files · 18.6 MB</span>
        </header>
        <div className="import-dialog-file">
          <FileImage aria-hidden="true" size={17} />
          <span>
            <strong>oak_albedo.png</strong>
            <small>4096 × 4096 · Texture</small>
          </span>
          <code>8.2 MB</code>
        </div>
        <div className="import-dialog-file">
          <FileImage aria-hidden="true" size={17} />
          <span>
            <strong>oak_normal.png</strong>
            <small>4096 × 4096 · Normal map</small>
          </span>
          <code>6.7 MB</code>
        </div>
        <div className="import-dialog-file">
          <FileImage aria-hidden="true" size={17} />
          <span>
            <strong>oak_roughness.png</strong>
            <small>4096 × 4096 · Texture</small>
          </span>
          <code>3.7 MB</code>
        </div>
      </div>

      <div className="import-dialog-section">
        <label className="import-dialog-field">
          <span>Destination</span>
          <input value={destination} onChange={(event) => setDestination(event.target.value)} />
        </label>
      </div>

      <div className="import-dialog-section import-dialog-options">
        <header>
          <strong>Texture options</strong>
          <span>Applied to compatible files</span>
        </header>
        <label>
          <input
            checked={generateMipmaps}
            type="checkbox"
            onChange={(event) => setGenerateMipmaps(event.target.checked)}
          />
          <span>
            <strong>Generate mipmaps</strong>
            <small>Create lower-resolution levels for runtime sampling.</small>
          </span>
        </label>
        <label>
          <input checked={srgb} type="checkbox" onChange={(event) => setSrgb(event.target.checked)} />
          <span>
            <strong>sRGB color</strong>
            <small>Treat color textures as gamma encoded.</small>
          </span>
        </label>
        <label className="import-dialog-field">
          <span>Compression</span>
          <select value={compression} onChange={(event) => setCompression(event.target.value)}>
            <option>Default (BC7)</option>
            <option>High Quality</option>
            <option>Uncompressed</option>
          </select>
        </label>
      </div>
    </UiDialog>
  );
}
