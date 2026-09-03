from pathlib import Path

path = Path('editor/src/renderer/src/uiLab/UiLab.tsx')
text = path.read_text()

import_anchor = "import { MenuBar } from '../layout/MenuBar';\n"
import_line = "import { UiLabContentCards } from './UiLabContentCards';\n"
if import_line not in text:
    if import_anchor not in text:
        raise SystemExit('UiLab import anchor not found')
    text = text.replace(import_anchor, import_anchor + import_line, 1)

section_anchor = '''        <LabSection\n          title="Navigation and containers"\n          description="Tabs, tree rows, panels, and ECS component containers."\n        >'''
card_section = '''        <LabSection\n          title="Content Browser cards"\n          description="Production asset cards with representative image, material, and model previews. Hover a card to inspect its asset-details surface."\n        >\n          <LabCard title="Asset cards + hover" caption="ContentAssetCard" wide>\n            <UiLabContentCards />\n          </LabCard>\n        </LabSection>\n\n'''
if card_section not in text:
    if section_anchor not in text:
        raise SystemExit('UiLab section anchor not found')
    text = text.replace(section_anchor, card_section + section_anchor, 1)

path.write_text(text)
