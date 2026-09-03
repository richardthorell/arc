from pathlib import Path
import re

panel = Path('editor/src/renderer/src/inspector/InspectorPanel.tsx')
text = panel.read_text()

import_anchor = "import type { AssetPickerItem, AssetThumbnailProvider } from './AssetPicker';\n"
if "import { AddComponentPicker } from './AddComponentPicker';" not in text:
    text = text.replace(import_anchor, import_anchor + "import { AddComponentPicker } from './AddComponentPicker';\n")

replacement = '''\n        <AddComponentPicker\n          snapshot={draft}\n          projectSchemas={projectSchemas}\n          onAdd={async (component, label) => {\n            const response = await command('component.add', { component });\n            if (!response.succeeded) {\n              setError(response.error || `Could not add ${label}`);\n              return false;\n            }\n            onStatus?.(`${label} added`);\n            await refresh();\n            return true;\n          }}\n        />'''

text, count = re.subn(
    r'\n\s*<details className="inspector-add-component">.*?</details>',
    replacement,
    text,
    count=1,
    flags=re.S,
)
if count != 1:
    raise SystemExit(f'Expected one Add Component details block, replaced {count}')
panel.write_text(text)

schemas = Path('editor/src/renderer/src/inspector/componentSchemas.ts')
schema_text = schemas.read_text()
anchor = "  projectComponent?: boolean;\n  fields: HostProjectFieldSchema[];"
if 'allowMultiple?: boolean;' not in schema_text:
    if anchor not in schema_text:
        raise SystemExit('HostProjectComponentSchema anchor not found')
    schema_text = schema_text.replace(anchor, "  projectComponent?: boolean;\n  allowMultiple?: boolean;\n  fields: HostProjectFieldSchema[];")
schemas.write_text(schema_text)
