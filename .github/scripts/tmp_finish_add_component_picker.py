from pathlib import Path

path = Path('editor/src/renderer/src/inspector/InspectorPanel.test.tsx')
text = path.read_text()
old = "await userEvent.click(screen.getByRole('button', { name: 'Gameplay / Gameplay Stats' }));"
new = "await userEvent.click(screen.getByRole('button', { name: 'Gameplay Stats' }));"
if old not in text:
    raise SystemExit('Expected old Add Component test assertion')
path.write_text(text.replace(old, new, 1))
