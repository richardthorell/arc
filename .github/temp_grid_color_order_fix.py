from pathlib import Path

path = Path('editor/src/renderer/src/viewport/ViewportPanel.tsx')
text = path.read_text()
start_marker = "  useEffect(() => {\n    let cancelled = false;\n    const applySnapshot = (next: import('../../../common/editorWorkflowTypes').EditorSettingsSnapshot | null) => {\n"
start = text.find(start_marker)
if start < 0:
    raise SystemExit('settings effect not found')
end_marker = "  }, [sendGridColor]);\n\n"
end = text.find(end_marker, start)
if end < 0:
    raise SystemExit('settings effect end not found')
end += len(end_marker)
effect = text[start:end]
text = text[:start] + text[end:]
anchor = "  }, [viewportActive, viewportId]);\n\n  const attachViewport = useCallback(async () => {\n"
if anchor not in text:
    raise SystemExit('sendGridColor/attach anchor not found')
text = text.replace(
    anchor,
    "  }, [viewportActive, viewportId]);\n\n" + effect + "  const attachViewport = useCallback(async () => {\n",
    1,
)
path.write_text(text)
print('Moved settings effect after sendGridColor declaration')
