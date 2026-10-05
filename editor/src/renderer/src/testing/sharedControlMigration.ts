const nativeControlPatterns = [
  { control: 'button', pattern: /<button\b/i },
  { control: 'select', pattern: /<select\b/i },
  {
    control: 'input',
    pattern: /<input\b(?![^>]*\btype=["'](?:file|hidden)["'])/i,
  },
  { control: 'textarea', pattern: /<textarea\b/i },
] as const;

export type NativeControlViolation = {
  control: (typeof nativeControlPatterns)[number]['control'];
  index: number;
};

export const findNativeControlViolations = (source: string): NativeControlViolation[] =>
  nativeControlPatterns.flatMap(({ control, pattern }) => {
    const match = pattern.exec(source);
    return match ? [{ control, index: match.index }] : [];
  });

export const formatNativeControlViolations = (
  fileName: string,
  violations: NativeControlViolation[],
): string =>
  violations.map(({ control }) => `${fileName} should use a shared Ui control instead of <${control}>`).join('\n');
