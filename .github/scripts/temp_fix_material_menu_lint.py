from pathlib import Path

path = Path('editor/src/renderer/src/material/MaterialGraphEditor.tsx')
text = path.read_text()
old = '''  const visibleSubcategories = nodeMenuCategory
    ? materialNodeSubcategoryOrder[nodeMenuCategory].filter((subcategory) =>
        availableNodes.some(
          (definition) => definition.category === nodeMenuCategory && definition.subcategory === subcategory,
        ),
      )
    : [];
  const visibleCategoryNodes =
    nodeMenuCategory && nodeMenuSubcategory
      ? availableNodes.filter(
          (definition) => definition.category === nodeMenuCategory && definition.subcategory === nodeMenuSubcategory,
        )
      : [];
'''
if old not in text:
    raise SystemExit('obsolete menu selector block not found')
path.write_text(text.replace(old, ''))
