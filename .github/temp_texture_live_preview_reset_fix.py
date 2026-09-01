from pathlib import Path

path = Path('editor/src/renderer/src/texture/TextureStage3Controls.test.tsx')
text = path.read_text()
if not text.startswith('// @vitest-environment jsdom'):
    text = '// @vitest-environment jsdom\n\n' + text
text = text.replace("screen.getByLabelText('Brightness')", "screen.getByRole('spinbutton', { name: 'Brightness' })")
path.write_text(text)
