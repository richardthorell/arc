from pathlib import Path

path = Path('editor/src/renderer/src/texture/TextureStage3Controls.test.tsx')
text = path.read_text()
if not text.startswith('// @vitest-environment jsdom'):
    path.write_text('// @vitest-environment jsdom\n\n' + text)
