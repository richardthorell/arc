import { useRef, useState, useEffect } from 'react';
import { Box } from 'lucide-react';
import { UiButton } from './UiButton';

/** File-dialog variant of the asset preview control; never registers a runtime asset. */
export function UiFileImagePicker({
  preview,
  hasValue = Boolean(preview),
  browse,
  onChange,
  onError,
}: {
  preview?: string;
  hasValue?: boolean;
  browse: () => Promise<string | null>;
  onChange: (image: string | null) => void;
  onError: (message: string) => void;
}) {
  const [loading, setLoading] = useState(false);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const choose = async () => {
    setLoading(true);
    try {
      const source = await browse();
      if (!source || !alive.current) return;
      const image = new Image();
      image.src = source;
      await image.decode();
      if (image.width * image.height > 64 * 1024 * 1024) throw new Error('Image dimensions exceed the limit');
      const canvas = document.createElement('canvas');
      canvas.width = 1280;
      canvas.height = 720;
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Image conversion is unavailable');
      const scale = Math.max(1280 / image.width, 720 / image.height);
      const width = 1280 / scale;
      const height = 720 / scale;
      context.drawImage(image, (image.width - width) / 2, (image.height - height) / 2, width, height, 0, 0, 1280, 720);
      const result = canvas.toDataURL('image/webp', 0.9);
      if (!result.startsWith('data:image/webp;')) throw new Error('WebP conversion is unavailable');
      if (alive.current) onChange(result);
    } catch (error) {
      if (alive.current) onError(String(error));
    } finally {
      if (alive.current) setLoading(false);
    }
  };
  return (
    <div className="project-thumbnail-picker">
      <div className="asset-picker-preview project-thumbnail-preview">
        {preview ? (
          <img src={preview} alt="Project thumbnail preview" />
        ) : (
          <span aria-label="Default ARC thumbnail">
            <Box size={28} /> ARC
          </span>
        )}
      </div>
      <div className="project-thumbnail-actions">
        <UiButton disabled={loading} onClick={() => void choose()}>
          {loading ? 'Preparing…' : hasValue ? 'Replace' : 'Browse'}
        </UiButton>
        <UiButton disabled={loading || !hasValue} onClick={() => onChange(null)}>
          Remove
        </UiButton>
      </div>
    </div>
  );
}
