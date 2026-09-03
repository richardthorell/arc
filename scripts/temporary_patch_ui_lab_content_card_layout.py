from pathlib import Path

ui_lab = Path('editor/src/renderer/src/uiLab/UiLab.tsx')
text = ui_lab.read_text()
text = text.replace("  wide = false,\n  children,\n}: {\n  title: string;\n  caption?: string;\n  wide?: boolean;\n  children: React.ReactNode;\n}) {\n  return (\n    <article className={`ui-lab-card ${wide ? 'ui-lab-card-wide' : ''}`}>", "  wide = false,\n  fullWidth = false,\n  children,\n}: {\n  title: string;\n  caption?: string;\n  wide?: boolean;\n  fullWidth?: boolean;\n  children: React.ReactNode;\n}) {\n  const widthClass = fullWidth ? 'ui-lab-card-full' : wide ? 'ui-lab-card-wide' : '';\n  return (\n    <article className={`ui-lab-card ${widthClass}`}>")
text = text.replace('<LabCard title="Asset cards + hover" caption="ContentAssetCard" wide>', '<LabCard title="Asset cards + hover" caption="ContentAssetCard" fullWidth>')
ui_lab.write_text(text)

cards = Path('editor/src/renderer/src/uiLab/UiLabContentCards.tsx')
text = cards.read_text()
text = text.replace("      <p>Hover any card to preview the production asset-details tooltip.</p>\n", "")
cards.write_text(text)

css = Path('editor/src/renderer/src/uiLab/uiLab.css')
text = css.read_text()
anchor = ".ui-lab-card-wide {\n  grid-column: span 2;\n}\n"
addition = anchor + "\n.ui-lab-card-full {\n  grid-column: 1 / -1;\n}\n"
if '.ui-lab-card-full {' not in text:
    if anchor not in text:
        raise SystemExit('wide card CSS anchor not found')
    text = text.replace(anchor, addition, 1)
css.write_text(text)

showcase_css = Path('editor/src/renderer/src/uiLab/uiLabContentCards.css')
text = showcase_css.read_text()
start = text.find('.ui-lab-content-card-showcase > p {')
if start != -1:
    end = text.find('\n}\n', start)
    if end == -1:
        raise SystemExit('helper paragraph CSS end not found')
    text = text[:start] + text[end + 3:]
showcase_css.write_text(text)
