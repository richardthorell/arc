import { FileImage, Upload } from 'lucide-react';
import { useState } from 'react';

import { UiButton, UiDialog, UiSelect, UiTextInput, UiToggleButton } from '../ui';

import './ImportDialog.css';

type ImportDialogProps = {
  onClose?: () => void;
  onImport?: () => void;
  preview?: boolean;
};

const compressionOptions = [
  { value: 'Default (BC7)', label: 'Default (BC7)' },
  { value: 'High Quality', label: 'High Quality' },
  { value: 'Uncompressed', label: 'Uncompressed' },
] as const;

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
          <UiTextInput value={destination} onChange={(event) => setDestination(event.target.value)} />
        </label>
      </div>

      <div className="import-dialog-section import-dialog-options">
        <header>
          <strong>Texture options</strong>
          <span>Applied to compatible files</span>
        </header>
        <div className="import-dialog-toggle-row">
          <UiToggleButton
            aria-label="Generate mipmaps"
            checked={generateMipmaps}
            onCheckedChange={setGenerateMipmaps}
          />
          <span>
            <strong>Generate mipmaps</strong>
            <small>Create lower-resolution levels for runtime sampling.</small>
          </span>
        </div>
        <div className="import-dialog-toggle-row">
          <UiToggleButton aria-label="sRGB color" checked={srgb} onCheckedChange={setSrgb} />
          <span>
            <strong>sRGB color</strong>
            <small>Treat color textures as gamma encoded.</small>
          </span>
        </div>
        <div className="import-dialog-field">
          <span>Compression</span>
          <UiSelect
            ariaLabel="Compression"
            options={compressionOptions}
            value={compression}
            onValueChange={setCompression}
          />
        </div>
      </div>
    </UiDialog>
  );
}
