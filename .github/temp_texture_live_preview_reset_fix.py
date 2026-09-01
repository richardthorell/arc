from pathlib import Path

path = Path('editor/src/renderer/src/texture/TextureStage3Controls.test.tsx')
text = path.read_text()
if not text.startswith('// @vitest-environment jsdom'):
    text = '// @vitest-environment jsdom\n\n' + text
text = text.replace("screen.getByLabelText('Brightness')", "screen.getByRole('spinbutton', { name: 'Brightness' })")
text = text.replace("expect(screen.getByRole('spinbutton', { name: 'Brightness' })).toHaveValue(0.6);", "expect((screen.getByRole('spinbutton', { name: 'Brightness' }) as HTMLInputElement).value).toBe('0.6');")
text = text.replace("expect(screen.getByRole('spinbutton', { name: 'Brightness' })).toHaveValue(0);", "expect((screen.getByRole('spinbutton', { name: 'Brightness' }) as HTMLInputElement).value).toBe('0');")
path.write_text(text)
