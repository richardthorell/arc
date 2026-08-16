export const arcThemes = [
  {
    id: 'arc-dark',
    label: 'Arc Dark',
  },
] as const;

export type ArcThemeId = (typeof arcThemes)[number]['id'];

export const defaultArcThemeId: ArcThemeId = 'arc-dark';

export function isArcThemeId(value: string | undefined): value is ArcThemeId {
  return arcThemes.some((theme) => theme.id === value);
}
