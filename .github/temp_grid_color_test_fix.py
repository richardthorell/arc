from pathlib import Path

path = Path('editor/src/renderer/src/viewport/ViewportPanel.tsx')
text = path.read_text()
old = "    void window.arc.settings.snapshot().then(applySnapshot);\n"
new = "    if (window.arc.settings) void window.arc.settings.snapshot().then(applySnapshot);\n"
if old not in text:
    raise SystemExit('settings snapshot anchor not found')
path.write_text(text.replace(old, new, 1))
print('Guarded optional settings bridge for isolated viewport tests')
