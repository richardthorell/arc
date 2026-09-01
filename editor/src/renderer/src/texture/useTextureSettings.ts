import { useEffect, useState } from 'react';

import { getTextureSettings, type TextureSettingsSnapshot } from './textureSettings';

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
    load();
    window.addEventListener('arc:texture-settings-changed', onChanged);
    return () => {
      active = false;
      window.removeEventListener('arc:texture-settings-changed', onChanged);
    };
  }, [generation, guid]);

  return { settings, error };
}
