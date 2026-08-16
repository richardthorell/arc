import { useEffect, useState } from 'react';

import { arcThemes, defaultArcThemeId, isArcThemeId } from '../themeRegistry';
import type { ArcThemeId } from '../themeRegistry';

function initialTheme(): ArcThemeId {
  const currentTheme = document.documentElement.dataset.theme;
  return isArcThemeId(currentTheme) ? currentTheme : defaultArcThemeId;
}

export function UiLabThemePicker() {
  const [theme, setTheme] = useState<ArcThemeId>(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  return (
    <label className="ui-lab-theme-picker">
      <span>Theme</span>
      <select aria-label="UI Lab theme" value={theme} onChange={(event) => setTheme(event.target.value as ArcThemeId)}>
        {arcThemes.map((option) => (
          <option key={option.id} value={option.id}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
