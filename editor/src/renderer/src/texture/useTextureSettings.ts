import { useEffect, useState } from 'react';

import { getTextureSettings, type TextureSettingsPatch, type TextureSettingsSnapshot } from './textureSettings';

export function useTextureSettings(guid: string | undefined, generation?: number) {
  const [settings, setSettings] = useState<TextureSettingsSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const load = () => {
      if (!guid) {
        setSettings(null);
        return;
      }
      void getTextureSettings(guid)
        .then((value) => {
          if (!active) return;
          setSettings(value);
          setError(null);
        })
        .catch((reason: unknown) => {
          if (active) setError(reason instanceof Error ? reason.message : 'Could not load texture settings');
        });
    };
    const onChanged = (event: Event) => {
      if ((event as CustomEvent<{ guid?: string }>).detail?.guid === guid) load();
    };
    const onPreview = (event: Event) => {
      const detail = (event as CustomEvent<{ guid?: string; patch?: TextureSettingsPatch }>).detail;
      if (detail?.guid !== guid || !detail.patch) return;
      setSettings((current) => (current ? { ...current, ...detail.patch } : current));
    };
    load();
    window.addEventListener('arc:texture-settings-changed', onChanged);
    window.addEventListener('arc:texture-settings-preview', onPreview);
    return () => {
      active = false;
      window.removeEventListener('arc:texture-settings-changed', onChanged);
      window.removeEventListener('arc:texture-settings-preview', onPreview);
    };
  }, [generation, guid]);

  return { settings, error };
}
